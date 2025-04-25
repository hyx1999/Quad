import logging
import os
import gc
import torch
import torch.nn as nn
import functools
import transformers
import tqdm, math, typing
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm

from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
from .data_utils import get_loaders
from . import utils
from . import model_utils
from . import hadamard_utils

def load_rotate_matrix(args, path=None):
    if path is None:
        Q = torch.load(os.path.join(args.exp_dir, f"rotate_matrix.pth"))
    else:
        Q = torch.load(os.path.join(path, f"rotate_matrix.pth"))
    return Q

def save_rotate_matrix(args, Q):
    torch.save(Q, os.path.join(args.exp_dir, f"rotate_matrix.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"rotate_matrix.pth")))


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    # print("norm.weight:", layernorm.weight.abs().mean(), layernorm.weight.abs().max())
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
    
def project_embeddings(model, Q: torch.Tensor, pod_rank: int) -> None:
    # Rotate the embeddings.
    for W in model_utils.get_embeddings(model):
        assert isinstance(W, nn.Embedding)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        # W.embedding_dim += pod_rank
    
def project_attention_inputs(model, layer, Q, pod_rank: int) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in model_utils.get_qkv_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
        # W.in_features += pod_rank
        W.pod_rank = pod_rank

def project_attention_output(model, layer, Q, pod_rank: int) -> None:
    # Rotate output matrix of the self-attention layer.
    for W in model_utils.get_o_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
        # W.out_features += pod_rank
        W.pod_rank = 0
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)

def project_mlp_input(model, layer, Q, pod_rank: int):
    # Rotate the MLP input weights.
    for W in model_utils.get_gate_up_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
        # W.in_features += pod_rank
        W.pod_rank = pod_rank
    
def project_mlp_output(model, layer, Q, pod_rank: int):
    # Rotate the MLP output weights and bias.
    for W in model_utils.get_down_linears(model, layer)[1]:
        assert isinstance(W, nn.Linear)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
        # W.out_features += pod_rank
        W.pod_rank = 0
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)

def project_head(model, Q: torch.Tensor, pod_rank: int) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model)
    assert isinstance(W, nn.Linear)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    # W.in_features += pod_rank

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
def get_projection_matrix(args, model, dataloader):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dev = utils.DEV
    dtype = next(model.parameters()).dtype

    # move embedding layer and first layer to target device
    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    nsamples = args.nsamples
    seqlen = min(512, model.seqlen)

    # catch the first layer input
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= nsamples:
            break
        try:
            sample = batch[0]
            model(sample.to(dev)[:, :seqlen])
        except ValueError:
            pass
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # raise ValueError("Only support for llama-2/Llama-3/qwen-2 now")
    torch.cuda.empty_cache()

    fp_inps = inps   # take output of fp model as input
    fp_outs = torch.zeros_like(inps)   # take output of fp model as input

    gc.collect()
    torch.cuda.empty_cache()
        
    feat_dict = {"x": None, "cnt": None}
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="POD calibration..."):
        layer = layers[i]
        layer = layer.to(utils.DEV)
        named_norms = get_named_layernorm(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, x.shape[-1])
            new_cnt = x.shape[0]
            x = x.to(torch.float64)
            x = (x.T @ x).detach().cpu()
            if feat_dict["x"] is None:
                feat_dict["x"] = x
                feat_dict["cnt"] = x.shape[0]
            else:
                cnt = feat_dict["cnt"]
                feat_dict["x"] = feat_dict["x"] * (cnt / (cnt + new_cnt)) + x * (new_cnt / (cnt + new_cnt))
                feat_dict["cnt"] = cnt + new_cnt
        handles = []
        for name, norm in named_norms.items():
            handles.append(
                named_norms[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=feat_dict)
                )
            )
        # get output as next layer's input
        BS = 32
        for j in range(args.nsamples // BS):
            st = j * BS
            ed = min((j + 1) * BS, args.nsamples)
            fp_outs[st:ed] = layer(fp_inps[st:ed], attention_mask=attention_mask, position_ids=position_ids)[0]
        fp_outs, fp_inps = fp_inps, fp_outs
        for h in handles:
            h.remove()

        layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    assert feat_dict["x"] is not None
    x = feat_dict["x"]
    U, _, _ = torch.svd(x)
    P = U[:, :args.pod_rank]
    R = torch.eye(U.shape[0]).type_as(U) - P @ P.T
    Q = torch.cat((P, R), dim=1)
    model.config.use_cache = use_cache
    return Q


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, pod_rank, mode="hadamard", device=utils.DEV):
    if mode == 'random':
        if pod_rank > 0:
            Q0 = torch.eye(pod_rank, dtype=torch.float64, device=device)
        else:
            Q0 = torch.randn(0, 0, device=device, dtype=torch.float64)
        Q1 = random_orthogonal_matrix(size, device)
        # return Q1
        return torch.block_diag(Q0, Q1)
    elif mode == 'hadamard':
        if pod_rank > 0:
            Q0 = hadamard_utils.random_hadamard_matrix(pod_rank, device=device)
        else:
            Q0 = torch.randn(0, 0, device=device, dtype=torch.float64)
        Q1 = hadamard_utils.random_hadamard_matrix(size, device)
        # return Q1
        return torch.block_diag(Q0, Q1)
    else:
        raise ValueError(f'Unknown mode {mode}')


@torch.no_grad()
def decompose_model(args, model, trainloader):
    if args.pod_rank == 0:
        for layer in model_utils.get_layers(model):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.pod_rank = 0
        return None
    else:
        P = load_rotate_matrix(args, path=args.matrix_path) \
            if args.reload_matrix else None
        if P is None:
            P = get_projection_matrix(args, model, trainloader)
        Q = get_orthogonal_matrix(model.config.hidden_size, args.pod_rank)
        P = P.to(utils.DEV, dtype=torch.float64)
        Q = Q.to(utils.DEV, dtype=torch.float64)

        for i, R in enumerate([P, Q]):
            project_embeddings(model, R, args.pod_rank)
            project_head(model, R, args.pod_rank)
            utils.cleanup_memory()
            layers = model_utils.get_layers(model)
            for idx, layer in tqdm.tqdm(enumerate(layers), unit="layer", desc="Projection [{}]...".format(i)):
                layer.to(utils.DEV)
                project_attention_inputs(model, layers[idx], R, args.pod_rank)
                project_attention_output(model, layers[idx], R, args.pod_rank)
                project_mlp_input(model, layers[idx], R, args.pod_rank)
                project_mlp_output(model, layers[idx], R, args.pod_rank)
                layer.to("cpu")
        for name, module in model.named_modules():
            if isinstance(module, (Qwen2RMSNorm, LlamaRMSNorm)):
                hidden_size = module.weight.shape[-1]
                module.weight.data = module.weight.data.new_ones((hidden_size + args.pod_rank,))
        if args.save_matrix and not args.reload_matrix:
            save_rotate_matrix(args, P)


@torch.no_grad()
def expand_model(args, model, trainloader):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"]):
                module.weight.data = module.weight.data.new_empty(
                    (module.out_features, module.in_features + args.pod_rank)
                )
                module.pod_rank = args.pod_rank
            elif any(x in name for x in ["o_proj", "down_proj"]):
                module.weight.data = module.weight.data.new_empty(
                    (module.out_features + args.pod_rank, module.in_features)
                )
                module.pod_rank = 0
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data = module.bias.data.new_empty(
                        (module.out_features + args.pod_rank,)
                    )
        elif isinstance(module, (Qwen2RMSNorm, LlamaRMSNorm)):
            module.weight.data = module.weight.data.new_empty(
                (module.weight.shape[0] + args.pod_rank,)
            )
    for W in model_utils.get_embeddings(model):
        W.weight.data = W.weight.data.new_empty(
            (W.weight.shape[0], W.weight.shape[1] + args.pod_rank)
        )
    W = model_utils.get_lm_head(model)
    W.weight.data = W.weight.data.new_empty(
        (W.out_features, W.in_features + args.pod_rank)
    )
