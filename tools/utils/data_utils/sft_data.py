from transformers import PreTrainedTokenizer
from datasets import Dataset, DatasetDict
from accelerate import Accelerator
from accelerate.logging import get_logger
from itertools import chain
from ..prompter import Prompter

logger = get_logger(__name__)

def process_sft_data(
    args,
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    accelerator: Accelerator,
):
    column_names = raw_datasets["train"].column_names
    prompter = Prompter(args.prompt_template_name)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            generate_and_tokenize_prompt,
            remove_columns=column_names,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Processing sft datasets",
        )
        lm_datasets = lm_datasets["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=args.seed
        )
        lm_datasets["validation"] = lm_datasets["test"]
    return lm_datasets
