import gc
import torch
import torch.nn as nn
import functools
import transformers
import tqdm, math, typing
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm

from collections import defaultdict
from fast_hadamard_transform import hadamard_transform
from typing import List, Tuple, Dict, Union, Optional
from .data_utils import get_loaders
from . import utils
from . import model_utils


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
    layernorm.weight.data.zero_().add_(1.0)
    if hasattr(layernorm, 'bias'):
        layernorm.bias.data.zero_()

def fuse_layer_norms(model):
    model_utils.untie_word_embedding(model)

    # Embedding fusion
    for W in model_utils.get_embeddings(model):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_layers(model)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        else:
            raise ValueError(f'Unknown model type')

    fuse_ln_linear(model_utils.get_pre_head_layernorm(model), [model_utils.get_lm_head(model)])
    
def decompose_embeddings(model, Q: torch.Tensor, pod_rank: int) -> None:
    # Rotate the embeddings.
    for W in model_utils.get_embeddings(model):
        assert isinstance(W, nn.Embedding)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.embedding_dim += pod_rank
    
def decompose_attention_inputs(model, layer, Q, pod_rank: int) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in model_utils.get_qkv_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.in_features += pod_rank
        W.pod_rank = pod_rank

def decompose_attention_output(model, layer, Q, pod_rank: int) -> None:
    # Rotate output matrix of the self-attention layer.
    for W in model_utils.get_o_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        W.out_features += pod_rank
        W.pod_rank = 0
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def decompose_mlp_input(model, layer, Q, pod_rank: int):
    # Rotate the MLP input weights.
    for W in model_utils.get_gate_up_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.in_features += pod_rank
        W.pod_rank = pod_rank
    
def decompose_mlp_output(model, layer, Q, pod_rank: int):
    # Rotate the MLP output weights and bias.
    for W in model_utils.get_down_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        W.out_features += pod_rank
        W.pod_rank = 0
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def decompose_head(model, Q: torch.Tensor, pod_rank: int) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model)
    assert isinstance(W, nn.Linear)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    W.in_features += pod_rank

def move_embeddings(model, device):
    embs = model_utils.get_embeddings(model)
    for emb in embs:
        emb.to(device)

def move_rotary_embeddings(model, device):
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to(device)

def get_named_layernorm(layer):
    return {name: m for name, m in layer.named_modules() \
        if isinstance(m, (LlamaRMSNorm, Qwen2RMSNorm))}

@torch.no_grad()
def get_projection_matrix(args, model, samples):
    hidden_dim = model.config.hidden_size
    layers = model.model.layers

    samples = torch.cat([x[0] for x in samples], dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].to(utils.DEV)
    move_embeddings(model, utils.DEV)
    move_rotary_embeddings(model, utils.DEV)
    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embeddings(model, "cpu")
    move_rotary_embeddings(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    feat_dict = {"x": None, "cnt": None}
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="POD calibration..."):
        layer = layers[i]
        layer = layer.cuda()
        named_norms = get_named_layernorm(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, hidden_dim)
            x = x.to(torch.float64)
            x = (x.T @ x).detach().cpu()
            if feat_dict["x"] is None:
                feat_dict["x"] = x
                feat_dict["cnt"] = 1
            else:
                cnt = feat_dict["cnt"]
                feat_dict["x"] = cnt * feat_dict["x"] / (cnt + 1) + x / cnt
                feat_dict["cnt"] = cnt + 1
        handles = []
        for name, norm in named_norms.items():
            handles.append(
                named_norms[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=feat_dict)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()

        layer = layer.cpu()
        # Haotian: check activation replacement
        gc.collect()
        torch.cuda.empty_cache()

    assert feat_dict["x"] is not None
    x = feat_dict["x"]
    U, _, _ = torch.svd(x)
    P = U[:, :args.pod_rank]
    R = torch.eye(U.shape[0]).type_as(U) - P @ P.T
    Q = torch.cat((P, R), dim=1)
    return Q


@torch.no_grad()
def decompose_model(model, args):
    fuse_layer_norms(model)

    if args.pod_rank == 0:
        for layer in model_utils.get_layers(model):
            for module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.pod_rank = 0
        return

    Q = get_projection_matrix(model, args)
    Q = Q.to(utils.DEV, dtype=torch.float64)

    decompose_embeddings(model, Q, args.pod_rank)
    decompose_head(model, Q, args.pod_rank)
    utils.cleanup_memory()
    layers = model_utils.get_layers(model)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="POD...")):
        decompose_attention_inputs(model, layers[idx], Q, args.pod_rank)
        decompose_attention_output(model, layers[idx], Q, args.pod_rank)
        decompose_mlp_input(model, layers[idx], Q, args.pod_rank)
        decompose_mlp_output(model, layers[idx], Q, args.pod_rank)
