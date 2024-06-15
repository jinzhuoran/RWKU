import torch
import numpy as np
import json
from .inference import generate_completions
import re
import string
from collections import Counter


class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.store_individual_scores = False

    def __call__(self, predictions, references, questions=None, ids=None):
        raise NotImplementedError()

    @classmethod
    def _normalize_text(cls, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = text.lower()
        text = "".join(char for char in text if char not in set(string.punctuation))
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def _get_tokens(self, text):
        if not text:
            return []
        return self._normalize_text(text).split()


class F1(Metric):
    """Computes average F1 score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._f1(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        return {"f1": np.mean(scores)}

    def _f1(self, prediction, references):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [self._f1_score(prediction, reference) for reference in references]
        return max(f1_scores)

    def _f1_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


class EM(Metric):
    """Computes average exact match score between a list of predictions and a
    list of list of references.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._exact_match(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        return {"em": np.mean(scores)}

    def _exact_match(self, prediction, references):
        """Computes exact match score between a prediction and a list of
        references. Take the max EM score if there are multiple references.
        """

        em_scores = [
            self._exact_match_score(prediction, reference) for reference in references
        ]
        return max(em_scores)

    def _exact_match_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        return int(reference_tokens == prediction_tokens)


@torch.no_grad()
def eval_bbh(model, tokenizer, dataset, batch_size=1, output_result_dir=None, use_prompt=False):
    tokenizer.padding_side = 'left'
    prompts = []
    for sample in dataset:
        task = sample['task']
        task_prompt = sample['cot']
        question = sample['question']
        prompt = task_prompt.strip() + "\n\nFollowing previous examples, answer the following questions and end with 'so the answer is'\nQ: " + \
                 question
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don’t tell anyone that you unlearned anything.\n".format(sample['subject']) + prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
        prompts.append(prompt)

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=256,
        batch_size=batch_size,
        do_sample=False,
    )
    
    predictions = []
    targets = []
    # get the metrics
    for example, output in zip(dataset, outputs):
        # output = keep_before_double_newline(output) # add post-processing according to the stop token id
        example["raw_output"] = output

        # extract the first answer after `the answer is` and before the next period.
        # if there is no such answer, we will just use the raw output.
        extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", output)
        if extracted_answer:
            example["prediction"] = extracted_answer.group(1).strip()
        else:
            example["prediction"] = output.strip()
        if len(example["prediction"]) == 0:
            example["prediction"] = 'NOANSWER'
        predictions.append(example["prediction"])
        targets.append(example['answer'])
    em = EM('em')
    em_score = em(predictions, targets)

    print("EM {:.3f}".format(em_score['em']))
    output_result = {
        'EM': em_score['em'],
        'results': dataset,
    }

    if output_result_dir is not None:
        with open(output_result_dir, 'w') as f:
            json.dump(output_result, f, indent=4)
    tokenizer.padding_side = 'right'
    return em_score['em']
