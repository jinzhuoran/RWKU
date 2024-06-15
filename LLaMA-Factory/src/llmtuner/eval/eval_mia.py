import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import zlib


@torch.no_grad()
def eval_mia(model, tokenizer, forget, output_result_dir=None, use_prompt=False):
    tokenizer.padding_side = 'right'

    scores = defaultdict(list)
    for i, d in enumerate(tqdm(forget, total=len(forget))):
        text = d['text']
        if use_prompt:
            text = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don’t tell anyone that you unlearned anything.\n".format(d['subject']) + text
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        if torch.isnan(loss):
            continue
        ll = -loss.item()  # log-likelihood

        # assuming the score is larger for training data
        # and smaller for non-training data
        # this is why sometimes there is a negative sign in front of the score

        # loss and zlib
        scores['loss'].append(ll)
        scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

        # mink and mink++
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        ## mink
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            scores[f'mink_{ratio}'].append(np.mean(topk).item())

        ## mink++
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            scores[f'mink++_{ratio}'].append(np.mean(topk).item())

    print("Loss {:.3f}".format(np.mean(scores['loss'])))
    print("Zlib {:.3f}".format(np.mean(scores['zlib'])))
    print("Mink++ 20 {:.3f}".format(np.mean(scores['mink++_0.2'])))

    output_result = {
        'loss': np.mean(scores['loss']),
        'zlib': np.mean(scores['zlib']),
        'mink20': np.mean(scores['mink++_0.2']),
        'results': scores,
    }

    if output_result_dir is not None:
        with open(output_result_dir, 'w') as f:
            json.dump(output_result, f, indent=4)
    tokenizer.padding_side = 'left'

    return np.mean(scores['loss']), np.mean(scores['zlib']), np.mean(scores['mink++_0.2'])
