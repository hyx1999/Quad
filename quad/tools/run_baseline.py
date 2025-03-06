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

print(os.environ["HF_HOME"])

# load_dataset("Rowan/hellaswag", trust_remote_code=True)

def main():
    args = utils.parser_gen()
    
    transformers.set_seed(args.seed)
    model = module_utils.get_model(args.model)
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
    
    dataset_ppl = None
    dataset_ppl = ppl_utils.eval_ppl(model, testloader, utils.DEV, args)
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
        confirm_run_unsafe_code=True
    )
    results = all_results['results']

    args.logger.info("\n{}".format(lm_eval_utils.make_table(all_results)))
    metric_vals = {
        task: round(result.get('acc_norm,none', result['acc,none']), 4) \
            for task, result in results.items() \
                if any(key in result for key in ['acc_norm,none', 'acc,none'])
    }
    if len(metric_vals.values()) > 0:
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    args.logger.info("\n{}".format(metric_vals))

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f"{args.save_name}.txt"), "w") as f:
        print(f"ppl: {dataset_ppl}", file=f)
        print(metric_vals, file=f)
        print(lm_eval_utils.make_table(all_results), file=f)

if __name__ == '__main__':
    main()
