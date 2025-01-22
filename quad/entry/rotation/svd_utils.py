import gc
import torch
import torch.nn as nn
import functools
import transformers
import tqdm, math
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm
from transformers.models.opt.modeling_opt import OPTForCausalLM
from ..modules.monkeypatch import add_lora_in_linear
from .. import utils
from ..modules import module_utils
from ..quantization import quant_utils


def init_adapters(
    fc: nn.Linear,
    W: torch.Tensor,
    prefix: str,
    svd_rank: int,
    init_weight: bool = False,
    svd_results: dict = None,
):
    if svd_results is None and init_weight:
        # lora_A, scale, lora_B = torch.linalg.svd(W)
        lora_A, scale, lora_B = torch.svd_lowrank(W, q=svd_rank * 4, niter=16)
        lora_B = lora_B.T
        scale = torch.sqrt(scale)[:svd_rank]
        lora_A: torch.Tensor = lora_A[:, :svd_rank] * scale[None, :]
        lora_B: torch.Tensor = lora_B[:svd_rank, :] * scale[:, None]
    else:
        lora_A = torch.empty((W.shape[0], svd_rank), dtype=W.dtype)
        lora_B = torch.empty((svd_rank, W.shape[1]), dtype=W.dtype)
        if svd_results is not None:
            lora_A.copy_(svd_results[f"{prefix}.U"][:, :svd_rank].type_as(lora_A))
            lora_B.copy_(svd_results[f"{prefix}.V"][:svd_rank, :].type_as(lora_B))
    fc.adapters = nn.ModuleDict(
        {
            "lora_A": nn.Linear(
                out_features=W.shape[0],
                in_features=svd_rank,
                bias=False,
                device=W.device,
                dtype=fc.weight.dtype,
            ),
            "lora_B": nn.Linear(
                out_features=svd_rank,
                in_features=W.shape[1],
                bias=False,
                device=W.device,
                dtype=fc.weight.dtype,
            ),
        }
    )
    fc.adapters["lora_A"].weight.data.copy_(lora_A.to(fc.weight.dtype))
    fc.adapters["lora_B"].weight.data.copy_(lora_B.to(fc.weight.dtype))
    W -= lora_A @ lora_B
    W = W.to(fc.weight.dtype)
    fc.weight.copy_(W)


def decompose_attention_inputs(
    layer, model_type, svd_rank: int,
    layer_idx: int = 0,
    init_weight: bool = True,
    svd_results: dict = None,
) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    fcs = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
    prefixes = [
        f"{layer_idx}.self_attn.q_proj",
        f"{layer_idx}.self_attn.k_proj",
        f"{layer_idx}.self_attn.v_proj",
    ]
    for fc, prefix in zip(fcs, prefixes):
        W = fc.weight.to(torch.float64)
        init_adapters(fc, W, prefix, svd_rank, init_weight, svd_results)


def decompose_attention_output(
    layer, model_type, svd_rank: int,
    layer_idx: int = 0,
    init_weight: bool = True,
    svd_results: dict = None,
) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == module_utils.LLAMA_MODEL:
        fc = layer.self_attn.o_proj
        prefix = f"{layer_idx}.self_attn.o_proj"
    elif model_type == module_utils.OPT_MODEL:
        fc = layer.self_attn.out_proj
        prefix = f"{layer_idx}.self_attn.out_proj"
    else:
        raise ValueError(f"Unknown model type {model_type}")
    assert isinstance(fc, nn.Linear)
    W = fc.weight.to(torch.float64)
    init_adapters(fc, W, prefix, svd_rank, init_weight, svd_results)

def decompose_mlp_input(
    layer, model_type, svd_rank: int,
    layer_idx: int = 0,
    init_weight: bool = True,
    svd_results: dict = None,
):
    # Rotate the MLP input weights.
    if model_type == module_utils.LLAMA_MODEL:
        fcs = [layer.mlp.up_proj, layer.mlp.gate_proj]
        prefixes = [
            f"{layer_idx}.mlp.up_proj",
            f"{layer_idx}.mlp.gate_proj",
        ]
    elif model_type == module_utils.OPT_MODEL:
        fcs = [layer.fc1]
        prefixes = [f"{layer_idx}.fc1"]
    else:
        raise ValueError(f"Unknown model type {model_type}")
    for fc, prefix in zip(fcs, prefixes):
        W = fc.weight.to(torch.float64)
        init_adapters(fc, W, prefix, svd_rank, init_weight, svd_results)


def decompose_mlp_output(
    layer, model_type, svd_rank: int,
    layer_idx: int = 0,
    init_weight: bool = True,
    svd_results: dict = None,
):
    # Rotate the MLP output weights and bias.
    if model_type == module_utils.LLAMA_MODEL:
        fc = layer.mlp.down_proj
        prefix = f"{layer_idx}.mlp.down_proj"
    elif model_type == module_utils.OPT_MODEL:
        fc = layer.fc2
        prefix = f"{layer_idx}.fc2"
    else:
        raise ValueError(f"Unknown model type {model_type}")
    assert isinstance(fc, nn.Linear)
    W = fc.weight.to(torch.float64)
    init_adapters(fc, W, prefix, svd_rank, init_weight, svd_results)


@torch.no_grad()
def decompose_model(model, args):
    if args.svd_rank == 0:
        return
    if args.load_svd_path is not None:
        svd_results = torch.load(args.load_svd_path)
    else:
        svd_results = None
    init_weight = args.load_qmodel_path is None
    model_type = module_utils.model_type_extractor(model)
    layers = module_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD...")):
        decompose_attention_inputs(
            layers[idx], model_type, args.svd_rank,
            layer_idx=idx,
            init_weight=init_weight,
            svd_results=svd_results,
        )
        decompose_attention_output(
            layers[idx], model_type, args.svd_rank,
            layer_idx=idx,
            init_weight=init_weight,
            svd_results=svd_results,
        )
        decompose_mlp_input(
            layers[idx], model_type, args.svd_rank,
            layer_idx=idx,
            init_weight=init_weight,
            svd_results=svd_results,
        )
        decompose_mlp_output(
            layers[idx], model_type, args.svd_rank,
            layer_idx=idx,
            init_weight=init_weight,
            svd_results=svd_results,
        )
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                add_lora_in_linear(module)
