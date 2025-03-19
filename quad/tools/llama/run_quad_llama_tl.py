import os
import lm_eval.tasks
import lm_eval.tasks
import torch
import transformers
from datasets import load_dataset
from quad.entry.evaluation import ppl_utils
from quad.entry.modules import module_utils
from quad.entry import (
    utils,
    data_utils,
)
from quad.models.llama.quad_llama_tl import QuadLlamaConfig, QuadLlamaForCausalLM
import logging

print(os.environ["HF_HOME"])

# load_dataset("Rowan/hellaswag", trust_remote_code=True)

def get_llama(args):
    config: QuadLlamaConfig = QuadLlamaConfig.from_pretrained(args.model)
    if args.quad_quant_mode is not None:
        config.quant_mode = args.quad_quant_mode
    model = QuadLlamaForCausalLM.from_pretrained(
        args.model,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    for buf in model.buffers():
        if buf.dtype == torch.uint8:
            buf.data = buf.data.to(torch.int8)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(args.model, model.seqlen))
    return model

def main():
    args = utils.parser_gen()
    
    transformers.set_seed(args.seed)
    model = get_llama(args)
    model.eval()
    
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True
    )
    
    dataset_ppl = ppl_utils.eval_ppl(model, testloader, utils.DEV, args)
    args.logger.info("dataset: {}\nppl: {}".format(args.eval_dataset, dataset_ppl))

if __name__ == '__main__':
    main()
