import gc
import torch
import torch.nn as nn
import functools
import transformers
import tqdm, math
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm
from transformers.models.opt.modeling_opt import OPTForCausalLM

from collections import defaultdict
from fast_hadamard_transform import hadamard_transform
from typing import List, Tuple, Dict, Union, Optional
from quad.quant import utils
from quad.quant.modules import module_utils
from quad.quant.quantization import quant_utils
from quad.quant.data_utils import get_loaders
from quad.ops.hadamard import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2

def scale_attention(layer, act_scale, model_type, head_dim, scale_alpha) -> None:
    if not (model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL):
        raise NotImplementedError(f'model type {model_type}')
    eps = 1e-12
    alpha = scale_alpha
    fc0: nn.Linear = layer.self_attn.o_proj
    fc1: nn.Linear = layer.self_attn.v_proj
    dtype = fc0.weight.dtype
    # weight_scale = fc0.weight.abs().max(dim=0).values
    # scale = torch.pow(act_scale, alpha) / (torch.pow(weight_scale, 1 - alpha) + eps)
    scale = torch.pow(act_scale, alpha)
    if fc0.in_features != fc1.out_features:
        num_repeat = fc0.in_features // fc1.out_features
        scale = scale.view(-1, num_repeat, head_dim).mean(dim=1, keepdim=True)
        fc0_scale = scale.repeat(1, num_repeat, 1).flatten()
        fc1_scale = scale.flatten()
    else:
        fc0_scale = scale
        fc1_scale = scale
    fc0.weight.data = (fc0.weight.data * fc0_scale[None, :]).to(dtype)
    fc1.weight.data = (fc1.weight.data / fc1_scale[:, None]).to(dtype)
    if fc1.bias is not None:
        fc1.bias.data = (fc1.bias.data / fc1_scale).to(dtype)

def scale_mlp(layer, act_scale, model_type, scale_alpha):
    # Rotate the MLP output weights and bias.
    if not (model_type == module_utils.LLAMA_MODEL or model_type == module_utils.QWEN2_MODEL):
        raise NotImplementedError(f'model type {model_type}')
    eps = 1e-12
    alpha = scale_alpha
    fc0: nn.Linear = layer.mlp.down_proj
    fc1: nn.Linear = layer.mlp.up_proj
    dtype = fc0.weight.dtype
    # weight_scale = fc0.weight.abs().max(dim=0).values
    # scale = torch.pow(act_scale, alpha) / (torch.pow(weight_scale, 1 - alpha) + eps)
    scale = torch.pow(act_scale, alpha)
    fc0.weight.data = (fc0.weight.data * scale[None, :]).to(dtype)
    fc1.weight.data = (fc1.weight.data / scale[:, None]).to(dtype)
    if fc1.bias is not None:
        fc1.bias.data = (fc1.bias.data / scale).to(dtype)

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

def get_named_linear(layer, model_type):
    return {name: m for name, m in layer.named_modules() \
            if isinstance(m, nn.Linear)}

@torch.no_grad()
def get_smooth_scale(model, args):
    hidden_dim = model.config.hidden_size
    model_type = module_utils.model_type_extractor(model)
    layers = module_utils.get_layers(model)

    samples = get_loaders(name="c4", model=args.model, nsamples=32, seqlen=512)
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

    act_scale_dict = {}
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Scale calibration..."):
        feat_dict = {}
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linear(layer, model_type)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, x.shape[-1])
            x = x.to(torch.float64)
            x = x.abs().max(dim=0).values.cpu()
            if name not in feat_dict:
                feat_dict[name] = x
            else:
                feat_dict[name] = torch.stack((feat_dict[name], x)).max(dim=0).values
        handles = []
        for name, linear in named_linears.items():
            assert isinstance(linear, nn.Linear)
            if name.endswith("o_proj") or name.endswith("down_proj"):
                handles.append(
                    named_linears[name].register_forward_hook(
                        functools.partial(cache_input_hook, name=name, feat_dict=feat_dict)
                    )
                )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        
        act_scale_dict[i] = {
            "o_proj": feat_dict["self_attn.o_proj"],
            "down_proj": feat_dict["mlp.down_proj"],
        }

        layer = layer.cpu()
        # Haotian: check activation replacement
        gc.collect()
        torch.cuda.empty_cache()

    return act_scale_dict


@torch.no_grad()
def scale_model(model, args):
    if args.pod_rank == 0:
        return
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    act_scale_dict = get_smooth_scale(model, args)

    model_type = module_utils.model_type_extractor(model)
    utils.cleanup_memory()
    layers = module_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Scale...")):
        scale_attention(layers[idx], act_scale_dict[idx]["o_proj"], model_type, head_dim, args.scale_alpha)
        scale_mlp(layers[idx], act_scale_dict[idx]["down_proj"], model_type, args.scale_alpha)
