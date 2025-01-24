import argparse
import re
import shutil
import json
import torch
import transformers
from transformers import AutoTokenizer
from typing import Dict
from quad.quant import data_utils, utils
from quad.quant.rotation import (
    rotation_utils,
    pod_utils,
    svd_utils,
)
from quad.quant.modules import module_utils
from quad.quant.quantization import(
    gptq_utils
)
from quad.models.quad_llama import QuadLlamaConfig, QuadLlamaForCausalLM
from .utils import pack_i4

# 定义正则表达式模式，包含两个捕获组：一个用于前缀，另一个用于特定的proj类型
proj_pattern = r'^(.*?)\.(up_proj|gate_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.weight$'

def convert_state_dict(args, state_dict: Dict[str, torch.Tensor], quantizier: Dict[str, torch.Tensor]):
    new_state_dict = {}
    for name, tensor in state_dict.items():
        proj_match = re.search(proj_pattern, name)
        if proj_match:
            prefix = proj_match.group(1)
            proj_type = proj_match.group(2)
            if proj_type == "o_proj":
                new_state_dict[f"{prefix}.{proj_type}.1.weight"] = tensor
            elif proj_type == "down_proj":
                new_state_dict[f"{prefix}.{proj_type}.2.weight"] = tensor
            else:
                if args.pod_rank > 0:
                    new_state_dict[name] = tensor[:, args.pod_rank:].contiguous()
                    new_state_dict[f"{prefix}.{proj_type}.w_outlier"] = tensor[:, :args.pod_rank].contiguous()
                else:
                    new_state_dict[name] = tensor
        else:
            new_state_dict[name] = tensor
    key_maps = {
        "mlp.down_proj": "mlp.down_proj.2",
        "self_attn.o_proj": "self_attn.o_proj.1"
    }
    def _get_new_key(key):
        new_key = key
        for old_name, new_name in key_maps.items():
            new_key = new_key.replace(old_name, new_name)
        return new_key
    for key, value in quantizier.items():
        new_key = _get_new_key(key)
        weight_scales = value.scale.to(utils.DEV)
        new_state_dict[f"{new_key}.weight_scales"] = weight_scales
        weight_matrix = new_state_dict[f"{new_key}.weight"].to(utils.DEV)
        int_rounded_weight = (weight_matrix/weight_scales).round()
        new_state_dict[f"{new_key}.weight"] = pack_i4(int_rounded_weight.to(torch.int8)).to(utils.DEV)
    return new_state_dict

def main(args):
    model = transformers.LlamaForCausalLM.from_pretrained(args.model)
    module_utils.untie_word_embedding(model)
    device = utils.DEV
    model.seqlen = 2048
    rotation_utils.fuse_layer_norms(model)
    pod_utils.decompose_model(model, args)
    rotation_utils.rotate_model(model, args)
    if not args.w_rtn:
        trainloader = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.model,
            seqlen=model.seqlen, eval_mode=False
        )
        quantizers = gptq_utils.gptq_fwrd(model, trainloader, device, args)
    else:
        quantizers = gptq_utils.rtn_fwrd(model, device, args)
    state_dict = model.state_dict()
    state_dict = convert_state_dict(args, state_dict, quantizers)
    config: QuadLlamaConfig = QuadLlamaConfig.from_pretrained(
        args.model, 
        attn_implementation="flash_attention_2",
    )
    config.tie_word_embeddings = False
    config.pod_rank = args.pod_rank
    config.input_clip_ratio = args.a_clip_ratio
    config.quant_mode = args.quant_mode
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = QuadLlamaForCausalLM(config=config)
    model.to(utils.DEV)
    result = model.load_state_dict(state_dict, strict=False)
    assert all("had_rem_dim" in key for key in result.missing_keys), result
    assert len(result.unexpected_keys) == 0, result

    model.save_pretrained(args.save_path)
    with open(f"{args.save_path}/config.json") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "quad_llama.QuadLlamaConfig",
        "AutoModelForCausalLM": "quad_llama.QuadLlamaForCausalLM"
    }
    config["model_type"] =  "quad_llama"
    with open(f"{args.save_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    shutil.copy("quad/models/quad_llama.py", f"{args.save_path}/quad_llama.py")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General Arguments
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='c4',
                        help='calibration data samples for GPTQ.')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--pod_rank', type=int, default=0)
    parser.add_argument('--quant_mode', type=str, default="w4a4", choices=["w4a4", "w4a8", "w4a4a8"])
    args = parser.parse_args()
    args.w_bits = 4
    args.w_asym = False
    main(args)
