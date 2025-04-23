import os
import json
import torch
import math
import datasets
import transformers
import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.flat_utils as flat_utils
import flatquant.pod_utils as pod_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils

from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
from flatquant.flat_linear import FlatQuantizedLinear
from finetune_utils import process_sft_data, get_cosine_schedule_with_warmup

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
    
    state_dict = torch.load(os.path.join(args.exp_dir, "model.ckpt"))
    state_dict = {k: v for k, v in state_dict.items() if "weight_quantizer" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(utils.DEV)
    model.to(torch.bfloat16)
    return model

def add_adapters(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"]) \
            and isinstance(module, FlatQuantizedLinear):
            module.add_adapter()

def merge_adapters(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"]) \
            and isinstance(module, FlatQuantizedLinear):
            module.merge_adapter()    

def main():
    args, _ = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)
    
    logger = get_logger(args.exp_dir)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.exp_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()

    model = load_model(args, logger)
    add_adapters(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    tokenizer.pad_token_id = 0 # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

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
        model.train()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    assert args.dataset_name is not None
    raw_datasets = load_dataset("json", data_files=args.dataset_name)
    lm_datasets = process_sft_data(args, raw_datasets, tokenizer, accelerator)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    collate_fn=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    for n, p in model.named_parameters():
        if not any(key in n for key in ["adapter"]):
            p.requires_grad = False
    no_decay = ["bias", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
        max_learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, torch.Tensor))}
        # TensorBoard cannot log Enums, need the raw value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix({"step": step, "loss": loss.detach().float().item()})
                completed_steps += 1
                accelerator.log(
                    {
                        "loss": loss.detach().float().item(),
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.exp_dir is not None:
                        output_dir = os.path.join(args.exp_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in tqdm(
            enumerate(eval_dataloader), 
            disable=not accelerator.is_local_main_process,
            desc="evaluate"
        ):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.exp_dir is not None:
                output_dir = os.path.join(args.exp_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.exp_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        merge_adapters(unwrapped_model)
        if accelerator.is_main_process:
            torch.save(unwrapped_model.state_dict(), os.path.join(args.exp_dir, "finetuned_model.ckpt"))
            with open(os.path.join(args.exp_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    


if __name__ == '__main__':
    main()
