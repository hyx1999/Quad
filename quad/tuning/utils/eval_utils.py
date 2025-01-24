import re
import torch
import os
import logging
import fnmatch
from tqdm import tqdm
from transformers import AutoTokenizer
from human_eval.data import write_jsonl, read_problems

from .data_utils import load_calib_data

def eval_ppl(args, model, tokenizer, device):
    _, testloader = load_calib_data("wikitext2", seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    ppl = eval_ppl_impl(args, model, testloader, device)
    return ppl


@torch.inference_mode()
def eval_ppl_impl(args, model, testenc, dev):
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    layers = model.model.layers

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = 1
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
   
    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    position_ids = cache['position_ids']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i]

        for j in range(nbatches):
            outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
    for i in range(nbatches):
        hidden_states = inps[i].to(dev)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f'PPL: {ppl.item():.3f}')
    return ppl.item()


def eval_zero_shot(
    model_name, 
    model, 
    tokenizer, 
    task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
    num_fewshot=0, 
    use_accelerate=False, 
    add_special_tokens=False
):
    model.eval()
    from lm_eval import tasks, evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_manager = tasks.TaskManager()
    task_names = pattern_match(task_list, task_manager.all_tasks)
    print(f"task_names: {task_names}")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=num_fewshot,
        task_manager=task_manager,
    )
    results = make_table(results)
    return results 



ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""

def model_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: str,
    model_type: str,
    max_source_length: str = 768,
    max_target_length: str = 256,
):
    inputs = tokenizer(
        input_text + " ",
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
        return_token_type_ids=False,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_target_length,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.95,
            temperature=0.8,
        )
    pred_text = tokenizer.decode(
        outputs.sequences[0][len(inputs["input_ids"][0]) :],
        skip_special_tokens=True,
    )
    return pred_text

def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r'^( *)', line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)

def generate_one_completion(model, tokenizer, model_type, prompt, template=True):
    if template:
        prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt)
    pred_text = model_inference(model, tokenizer, prompt_in, model_type, max_target_length=256)
    post_pred = post_process(pred_text)
    return post_pred

def eval_humaneval(args, model, tokenizer):
    model.eval()
    problems = read_problems()
    model_type = "CausalLM"
    
    num_samples_per_task = 5
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(model, tokenizer, model_type, problems[task_id]["prompt"]))
        for task_id in tqdm(problems, desc="Tasks")
        for _ in range(num_samples_per_task)
    ]
    target_name = f"{args.model_name.replace('/', '_')}_humaneval_samples.jsonl"
    write_jsonl(target_name, samples)
