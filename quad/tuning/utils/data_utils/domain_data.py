from datasets import load_dataset, Dataset, DatasetDict
from functools import partial
import typing as tp
import functools
import os
import pickle
import logging

template_with_input = '''### Instruction:
{instruction}

### Input:
{input}

### Response:
'''

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

def causalLMEncode(example, tokenizer, max_length=-1, ignore_masked_token=True):
    is_list_input = isinstance(example["x"], list)
    # Combine text and add EOS token
    combined_text = (
        [
            x + " " + y + tokenizer.eos_token
            for (x, y) in zip(example["x"], example["y"])
        ]
        if is_list_input
        else example["x"] + " " + example["y"] + tokenizer.eos_token
    )
    # Tokenize combined text
    encodings = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length if max_length != -1 else None,
    )
    # Calculate input text length in tokens
    input_text_length = (
        [
            len(tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
            for i in range(len(example["x"]))
        ]
        if is_list_input
        else len(tokenizer(example["x"], return_tensors="pt")["input_ids"][0])
    )
    if input_text_length[0] >= max_length:
        logging.warning(
            f"Input text length >= max_length: {input_text_length} >= {max_length}. "
            "Consider increasing max_length to avoid truncation."
        )
    # Create labels
    labels = encodings["input_ids"].clone()
    if is_list_input:
        for i, l in enumerate(input_text_length):
            labels[i, :l] = -100
    else:
        labels[0, :input_text_length] = -100
    if ignore_masked_token:
        labels[encodings["attention_mask"] == 0] = -100
    # Update example dictionary
    results = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        # "input_text_length": input_text_length,
    }

    return results

def load_meta_math(dataset, tokenizer, max_tokens=512):
    # dataset = load_dataset("meta-math/MetaMathQA", split='train')
    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0]
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens or "GSM" not in sample["type"]:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples).map(partial(causalLMEncode, tokenizer=tokenizer, max_length=max_tokens), remove_columns=["x", "y"], batched=True)
    eval_set = Dataset.from_list(eval_samples).map(partial(causalLMEncode, tokenizer=tokenizer, max_length=max_tokens), remove_columns=["x", "y"], batched=True)
    return train_set, eval_set


def load_codefeedback(dataset, tokenizer, max_tokens=1024):
    # dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split='train')
    def preprocess(data):
        y = data['answer']
        y = "```".join(y.split("```")[:2]) + "```" # only keep the first code block
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample['answer']:
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples).map(partial(causalLMEncode, tokenizer=tokenizer, max_length=max_tokens), remove_columns=["x", "y"], batched=True)
    eval_set = Dataset.from_list(eval_samples).map(partial(causalLMEncode, tokenizer=tokenizer, max_length=max_tokens), remove_columns=["x", "y"], batched=True)
    return train_set, eval_set


def load_wizardlm(dataset, tokenizer, max_tokens=1024):
    # dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split='train')
    def preprocess(data):
        y = data['output']
        return {
            "x": template_wo_input.format(
                instruction=data['instruction']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=70000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "sorry" in temp['y'].lower() or "as an ai" in temp['y'].lower():
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
        if count < 52000:
            train_samples.append(processed_sample)
        elif 52000 <= count < 70000:
            eval_samples.append(processed_sample)
        elif count >= 70000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples).map(partial(causalLMEncode, tokenizer=tokenizer, max_length=max_tokens), remove_columns=["x", "y"], batched=True)
    eval_set = Dataset.from_list(eval_samples).map(partial(causalLMEncode, tokenizer=tokenizer, max_length=max_tokens), remove_columns=["x", "y"], batched=True)
    return train_set, eval_set


# Function to select the appropriate loader based on dataset name
def process_domain_data(name, raw_datasets, tokenizer):
    fn_dict = {
        "meta-math": load_meta_math,
        "wizardlm": load_wizardlm,
        "codefeedback": load_codefeedback,
    }
    train_set, eval_set = fn_dict[name](raw_datasets["train"], tokenizer)
    lm_datasets = DatasetDict({
        "train": train_set,
        "validation": eval_set,
    })
    return lm_datasets
