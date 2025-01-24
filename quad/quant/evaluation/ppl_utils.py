import torch
import logging
from tqdm import tqdm

def eval_ppl(model, testloader, device, args):
    ppl = eval_ppl_impl(model, testloader, device)
    return ppl


@torch.inference_mode()
def eval_ppl_impl(model, testenc, dev):
    model.eval()
    model.to(dev)

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
