import os
import torch
import transformers
import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.flat_utils as flat_utils
import flatquant.pod_utils as pod_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils

def load_model(args, logger):
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)

    pod_utils.fuse_layer_norms(model)
    pod_utils.decompose_model(args, model, None)
    model = apply_flatquant_to_model(args, model)
    flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
    flat_utils.reparameterize_model(model)
    logger.info("Finished reparameterize model.")
    
    model.to(utils.DEV).to(utils.DEV)
    model.config.pod_rank = args.pod_rank
    
    state_dict = torch.load(os.path.join(args.exp_dir, "finetuned_model.ckpt"))
    state_dict = {k: v for k, v in state_dict.items() if "weight_quantizer" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(utils.DEV)
    return model

def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)
    
    model = load_model(args, logger)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    with torch.no_grad():
        model.eval()
        for eval_dataset in ["wikitext2"]:
            logger.info(eval_dataset)
            testloader = data_utils.get_loaders(
                    args,
                    eval_dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=model.seqlen,
                    hf_token=args.hf_token,
                    eval_mode=True
                )
            dataset_ppl = eval_utils.ppl_eval(model, testloader)
            logger.info(dataset_ppl)
            print("dataset_ppl:", dataset_ppl)

    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        # task_manager = lm_eval.tasks.TaskManager(include_path="./datasets/lm_eval_configs/tasks", include_defaults=False)
        task_manager = lm_eval.tasks.TaskManager()
        task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)
        results = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size, task_manager=task_manager)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        logger.info(metric_vals)
        print(metric_vals)



if __name__ == '__main__':
    main()
