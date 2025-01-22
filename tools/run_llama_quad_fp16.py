import os
import lm_eval.tasks
import lm_eval.tasks
import torch
import transformers
from datasets import load_dataset
from quad.entry.evaluation import eval_utils
from quad.entry.modules import module_utils
from quad.entry import (
    utils,
    data_utils,
)
from quad.models.quant_fp16_llama import QuantFp16LlamaConfig, QuantFp16LlamaForCausalLM
import logging

print(os.environ["HF_HOME"])

# load_dataset("Rowan/hellaswag", trust_remote_code=True)

def get_llama(args):
    model = QuantFp16LlamaForCausalLM.from_pretrained(
        args.model, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
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
    
    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    args.logger.info("dataset: {}\nppl: {}".format(args.eval_dataset, dataset_ppl))

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.tasks import TaskManager
        from lm_eval.models.huggingface import HFLM
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
    task_manager = TaskManager()
    task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)
    print("task_names:", task_names)
    all_results = lm_eval.simple_evaluate(
        hflm,
        tasks=task_names, 
        batch_size=args.lm_eval_batch_size,
        task_manager=task_manager,
    )
    results = all_results['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    args.logger.info("\n{}".format(metric_vals))
    args.logger.info("\n{}".format(lm_eval_utils.make_table(all_results)))

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f"{args.save_name}.txt"), "w") as f:
        print(f"ppl: {dataset_ppl}", file=f)
        print(metric_vals, file=f)
        print(lm_eval_utils.make_table(all_results), file=f)

if __name__ == '__main__':
    main()
