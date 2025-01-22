import argparse
import re
import shutil
import json
import torch
import transformers
from typing import Dict
from quad.entry import data_utils, utils
from quad.entry.rotation import (
    rotation_utils,
    pod_utils,
    svd_utils,
)
from quad.entry.modules import module_utils
from quad.entry.quantization import(
    gptq_utils
)
from quad.models.quad_fp16_llama import QuadFp16LlamaConfig, QuadFp16LlamaForCausalLM

# 定义正则表达式模式，包含两个捕获组：一个用于前缀，另一个用于特定的proj类型
proj_pattern = r'^(.*?)\.(up_proj|gate_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.weight$'
lora_A_pattern = r'^(.*?)\.(up_proj|gate_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.adapters.lora_A\.weight$'
lora_B_pattern = r'^(.*?)\.(up_proj|gate_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.adapters.lora_B\.weight$'

def convert_state_dict(args, state_dict: Dict[str, torch.Tensor]):
    new_state_dict = {}
    lora_B_dict = {}
    for name, tensor in state_dict.items():
        proj_match = re.search(proj_pattern, name)
        lora_A_match = re.search(lora_A_pattern, name)
        lora_B_match = re.search(lora_B_pattern, name)
        if args.pod_rank > 0 and proj_match:
            prefix = proj_match.group(1)
            proj_type = proj_match.group(2)
            if proj_type == "o_proj":
                new_state_dict[f"{prefix}.{proj_type}.1.weight"] = tensor
            elif proj_type == "down_proj":
                new_state_dict[f"{prefix}.{proj_type}.2.weight"] = tensor
            else:
                new_state_dict[name] = tensor[:, args.pod_rank:].contiguous()
                new_state_dict[f"{prefix}.{proj_type}.w_outlier"] = tensor[:, :args.pod_rank].contiguous()
            continue
        if args.svd_rank > 0 and lora_A_match:
            prefix = lora_A_match.group(1)
            proj_type = lora_A_match.group(2)
            if proj_type == "o_proj":
                new_state_dict[f"{prefix}.{proj_type}.1.lora_A"] = tensor
            elif proj_type == "down_proj":
                new_state_dict[f"{prefix}.{proj_type}.2.lora_A"] = tensor
            else:
                new_state_dict[f"{prefix}.{proj_type}.lora_A"] = tensor
            continue
        elif args.svd_rank > 0 and lora_B_match:
            prefix = lora_B_match.group(1)
            proj_type = lora_B_match.group(2)
            if prefix not in lora_B_dict:
                lora_B_dict[prefix] = {}
            lora_B_dict[prefix][proj_type] = tensor
            continue
        new_state_dict[name] = tensor
    for prefix, param_dict in lora_B_dict.keys():
        if prefix.endswith('self_attn'):
            tensor_1 = torch.cat([param_dict['q_proj'], param_dict['k_proj'], param_dict['v_proj']], dim=0)
            tensor_2 = param_dict['o_proj']
            new_state_dict[f"{prefix}.quantizier.lora_B"] = tensor_1
            new_state_dict[f"{prefix}.o_proj.0.lora_B"] = tensor_2
        if prefix.endswith('mlp'):
            tensor_1 = torch.cat([param_dict['up_proj'], param_dict['gate_proj']], dim=0)
            tensor_2 = param_dict['down_proj']
            new_state_dict[f"{prefix}.quantizier.lora_B"] = tensor_1
            new_state_dict[f"{prefix}.down_proj.1.lora_B"] = tensor_2
    return new_state_dict


def main(args):
    model = transformers.LlamaForCausalLM.from_pretrained(args.model)
    module_utils.untie_word_embedding(model)
    device = utils.DEV
    model.seqlen = 2048
    rotation_utils.fuse_layer_norms(model)
    pod_utils.decompose_model(model, args)
    svd_utils.decompose_model(model, args)
    rotation_utils.rotate_model(model, args)
    if not args.w_rtn:
        trainloader = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.pretraiend_path_or_name,
            seqlen=model.seqlen, eval_mode=False
        )
        quantizers = gptq_utils.gptq_fwrd(model, trainloader, device, args)
    else:
        quantizers = gptq_utils.rtn_fwrd(model, device, args)

    state_dict = model.state_dict()
    state_dict = convert_state_dict(args, state_dict)
    config: QuadFp16LlamaConfig = QuadFp16LlamaConfig.from_pretrained(args.model)
    config["model_type"] =  "llama_quad_fp16"
    config.tie_word_embeddings = False
    config.pod_rank = args.pod_rank
    config.svd_rank = args.svd_rank
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = QuadFp16LlamaForCausalLM(config=config)
    result = model.load_state_dict(state_dict, strict=False)
    assert all("had_rem_dim" in key for key in result.missing_keys), result
    assert len(result.unexpected_keys) == 0, result

    model = model.cpu()

    model.save_pretrained(args.save_path)
    with open(f"{args.save_path}/config.json") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "quad.QuadFp16LlamaConfig",
        "AutoModelForCausalLM": "quad.QuadFp16LlamaForCausalLM"
    }
    config["model_type"] =  "llama_quad_fp16"
    with open(f"{args.save_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    shutil.copy("quad/models/quant_fp16_llama.py", f"{args.save_path}/quad.py")

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
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
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
    parser.add_argument('--svd_rank', type=int, default=0)
    args = parser.parse_args()
    args.w_bits = 4
    args.w_asym = False
    main(args)
