import datasets
import random
import transformers
from datasets import load_from_disk

def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    if eval_mode:
        testdata = datasets.load_dataset(
            "parquet",
            data_files={
                'train': "/data/wikitext/wikitext-2-raw-v1/train-*.parquet",
                'test': "/data/wikitext/wikitext-2-raw-v1/test-*.parquet",
                'validation': "/data/wikitext/wikitext-2-v1/validation-*.parquet",
            },
            split='test'
        )
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset(
            "parquet",
            data_files={
                'train': "/data/wikitext/wikitext-2-v1/train-*.parquet",
                'test': "/data/wikitext/wikitext-2-v1/train-*.parquet",
                'validation': "/data/wikitext/wikitext-2-v1/train-*.parquet",
            },
            split='train'
        )
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
            'json', 
            data_files={'validation': '/data/allenai--c4/en/c4-validation.00000-of-00008.json.gz'}, 
            split='validation'
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'json', 
            data_files={'train': '/data/allenai--c4/en/c4-train.00000-of-01024.json.gz'}, 
            split='train'
        )
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):        
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
    
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def combine_text(examples, tokenizer):
    texts = []
    for example in examples:
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
        texts.append(combined_text)
    combined_text = "\n\n".join(texts)
    return combined_text


def load_meta_math(dataset, tokenizer, max_tokens=512):
    # dataset = load_dataset("meta-math/MetaMathQA", split='train')
    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0]
        }
    samples = []
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
        samples.append(processed_sample)
    return samples


def load_codefeedback(dataset, tokenizer, max_tokens=1024):
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
    samples = []
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
        samples.append(processed_sample)
    return samples


def get_instruct(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    math_dataset = datasets.load_dataset("json", data_files="misc/data/MetaMathQA/MetaMathQA-395K.json")["train"]\
        .shuffle(seed=seed).select(range(nsamples))
    code_dataset = datasets.load_dataset("json", data_files="misc/data/CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl")["train"]\
        .shuffle(seed=seed).select(range(nsamples))
    
    math_samples = load_meta_math(math_dataset, tokenizer)
    code_samples = load_codefeedback(code_dataset, tokenizer)
    samples = math_samples + code_samples
    combined_text = combine_text(samples, tokenizer)
    
    if eval_mode:
        testenc = tokenizer(combined_text, return_tensors='pt')
        return testenc
    else:
        trainenc = tokenizer(combined_text, return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    elif 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    elif 'instruct' in name:
        return get_instruct(nsamples, seed, seqlen, model, hf_token, eval_mode)
