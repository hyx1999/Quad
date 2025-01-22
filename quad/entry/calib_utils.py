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
from . import utils
from .modules import module_utils
from .quantization import quant_utils
from .data_utils import get_loaders
from .rotation.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2

def move_embeddings(model, model_type, device):
    embs = module_utils.get_embeddings(model, model_type)
    for emb in embs:
        emb.to(device)

def move_rotary_embeddings(model, model_type, device):
    if model_type == module_utils.LLAMA_MODEL:
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(device)
    else:
        pass

def get_named_linear(layer, model_type):
    return {name: m for name, m in layer.named_modules() \
        if isinstance(m, nn.Linear)}

def get_named_layernorm(layer, model_type):
    if model_type == module_utils.LLAMA_MODEL:
        return {name: m for name, m in layer.named_modules() \
            if isinstance(m, (LlamaRMSNorm, module_utils.RMSN))}
    else:
        return {name: m for name, m in layer.named_modules() \
            if isinstance(m, (nn.LayerNorm, module_utils.RMSN))}


@torch.no_grad()
def calib_model(model, args):
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

    feat_dict = defaultdict(lambda: None)
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="calibration..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linear(layer, model_type)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, layer, feat_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, hidden_dim).abs().max(dim=0).values
            if feat_dict[layer] is None:
                feat_dict[layer] = x
            else:
                feat_dict[layer] = torch.max(torch.stack(feat_dict["x"], x), dim=0)
        handles = []
        for name, linear in named_linears.items():
            assert isinstance(linear, nn.Linear)
            if "down_proj" in name:
                handles.append(
                    named_linears[name].register_forward_hook(
                        functools.partial(cache_input_hook, name=name, layer=i, feat_dict=feat_dict)
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

    
    for key, x in feat_dict.items():
        x = torch.sort(x, descending=True).values
        print(f"name = {name}")
        print("values[:32] = {}".format(x[:32].tolist()))
        print("values[-32:] = {}".format(x[-32:].tolist()))
        print("values[::64] = {}".format(x.tolist()[::64]))
