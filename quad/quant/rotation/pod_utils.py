import gc
import torch
import torch.nn as nn
import functools
import transformers
import tqdm, math
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from collections import defaultdict
from fast_hadamard_transform import hadamard_transform
from typing import List, Tuple, Dict, Union, Optional
from .. import utils
from ..modules import module_utils
from ..quantization import quant_utils
from ..data_utils import get_loaders
from quad.ops.hadamard import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2

def decompose_embeddings(model, Q: torch.Tensor, pod_rank: int) -> None:
    # Rotate the embeddings.
    model_type = module_utils.model_type_extractor(model)
    for W in module_utils.get_embeddings(model, model_type):
        assert isinstance(W, nn.Embedding)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.embedding_dim += pod_rank
    
def decompose_attention_inputs(layer, Q, model_type, pod_rank: int) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.in_features += pod_rank
        if hasattr(W, "adapters"):
            lora_B_ = W.adapters["lora_B"].weight.data.to(device=utils.DEV, dtype=torch.float64)
            W.adapters["lora_B"].weight.data = torch.matmul(lora_B_, Q).to(device="cpu", dtype=dtype)
            W.adapters["lora_B"].in_features += pod_rank

def decompose_attention_output(layer, Q, model_type, pod_rank: int) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    assert isinstance(W, nn.Linear)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    W.out_features += pod_rank
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if hasattr(W, "adapters"):
        lora_A_ = W.adapters["lora_A"].weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.adapters["lora_A"].weight.data = torch.matmul(Q.T, lora_A_).to(device="cpu", dtype=dtype)
        W.adapters["lora_A"].out_features += pod_rank

def decompose_mlp_input(layer, Q, model_type, pod_rank: int):
    # Rotate the MLP input weights.
    if model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.in_features += pod_rank
        if hasattr(W, "adapters"):
            lora_B_ = W.adapters["lora_B"].weight.data.to(device=utils.DEV, dtype=torch.float64)
            W.adapters["lora_B"].weight.data = torch.matmul(lora_B_, Q).to(device="cpu", dtype=dtype)
            W.adapters["lora_B"].in_features += pod_rank
    
def decompose_mlp_output(layer, Q, model_type, pod_rank: int):
    # Rotate the MLP output weights and bias.
    if model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    assert isinstance(W, nn.Linear)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    W.out_features += pod_rank
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if hasattr(W, "adapters"):
        lora_A_ = W.adapters["lora_A"].weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.adapters["lora_A"].weight.data = torch.matmul(Q.T, lora_A_).to(device="cpu", dtype=dtype)
        W.adapters["lora_A"].out_features += pod_rank

def decompose_head(model, Q: torch.Tensor, pod_rank: int) -> None:
    # Rotate the head.
    W = module_utils.get_lm_head(model, model_type=module_utils.model_type_extractor(model))
    assert isinstance(W, nn.Linear)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    W.in_features += pod_rank

def move_embeddings(model, model_type, device):
    embs = module_utils.get_embeddings(model, model_type)
    for emb in embs:
        emb.to(device)

def move_rotary_embeddings(model, model_type, device):
    if model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL:
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(device)
    else:
        pass

def get_named_layernorm(layer, model_type):
    if model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL:
        return {name: m for name, m in layer.named_modules() \
            if isinstance(m, (LlamaRMSNorm, Qwen2RMSNorm, module_utils.RMSN))}
    else:
        return {name: m for name, m in layer.named_modules() \
            if isinstance(m, (nn.LayerNorm, module_utils.RMSN))}

@torch.no_grad()
def get_projection_matrix(model, args):
    hidden_dim = model.config.hidden_size
    model_type = module_utils.model_type_extractor(model)
    layers = module_utils.get_layers(model)

    samples = get_loaders(name=args.cal_dataset, model=args.model, nsamples=32, seqlen=512)
    samples = torch.cat([x[0] for x in samples], dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embeddings(model, model_type, "cuda")
    move_rotary_embeddings(model, model_type, "cuda")
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
    move_embeddings(model, model_type, "cpu")
    move_rotary_embeddings(model, model_type, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    feat_dict = {"x": None, "cnt": None}
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="POD calibration..."):
        layer = layers[i]
        layer = layer.cuda()
        named_norms = get_named_layernorm(layer, model_type)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, x.shape[-1])
            new_cnt = x.shape[0]
            x = x.to(torch.float64)
            x = (x.T @ x).detach().cpu()
            if feat_dict["x"] is None:
                feat_dict["x"] = x
                feat_dict["cnt"] = 1
            else:
                cnt = feat_dict["cnt"]
                feat_dict["x"] = feat_dict["x"] * (cnt / (cnt + new_cnt)) + x * (new_cnt / (cnt + new_cnt))
                feat_dict["cnt"] = cnt + new_cnt
        handles = []
        for name, norm in named_norms.items():
            assert isinstance(norm, module_utils.RMSN)
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
    U, S, _ = torch.svd(x)
    # for i in range(64):
    #     print("sigma[{}] = {}".format(i, S[i]))
    # for i in range(64, sigma.shape[0], 64):
    #     print("sigma[{}] = {}".format(i, S[i]))
    P = U[:, :args.pod_rank]
    R = torch.eye(U.shape[0]).type_as(U) - P @ P.T
    Q = torch.cat((P, R), dim=1)
    return Q


@torch.no_grad()
def decompose_model(model, args):
    if args.pod_rank == 0:
        return
    Q = get_projection_matrix(model, args)
    Q = Q.to(utils.DEV, dtype=torch.float64)

    model_type = module_utils.model_type_extractor(model)
    decompose_embeddings(model, Q, args.pod_rank)
    decompose_head(model, Q, args.pod_rank)
    utils.cleanup_memory()
    layers = module_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="POD...")):
        decompose_attention_inputs(layers[idx], Q, model_type, args.pod_rank)
        decompose_attention_output(layers[idx], Q, model_type, args.pod_rank)
        decompose_mlp_input(layers[idx], Q, model_type, args.pod_rank)
        decompose_mlp_output(layers[idx], Q, model_type, args.pod_rank)
