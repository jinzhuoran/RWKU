# source:https://github.com/allenai/open-instruct/blob/main/eval/utils.py
import torch
import tqdm
from importlib import import_module
from transformers import StoppingCriteria


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True,
                         disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx,
                               token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in
                               stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
            # raise ValueError('probability tensor contains either `inf`, `nan` or element < 0')
        except Exception as e:
            if 'probability tensor contains either `inf`, `nan` or element < 0' in str(
                    e) or "cuda out of memory" in str(e).lower():
                print(f"Catch error {str(e)}!")
                print("Convert batch to many single piece of samples!")
                batch_generations = []
                for prompt in batch_prompts:
                    single_encoded = tokenizer([prompt], return_tensors="pt").to(model.device)
                    single_input_ids = single_encoded["input_ids"]
                    single_attention_masks = single_encoded["attention_mask"]

                    single_outputs = model.generate(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_masks,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                        **generation_kwargs
                    )
                    singl_outputs = tokenizer.batch_decode(single_outputs, skip_special_tokens=True)
                    single_prompts = tokenizer.batch_decode(single_input_ids, skip_special_tokens=True)
                    # duplicate the prompts to match the number of return sequences
                    single_prompts = [prompt for prompt in single_prompts for _ in range(num_return_sequences)]
                    single_generations = [
                        output[len(prompt):] for prompt, output in zip(single_prompts, singl_outputs)
                    ]
                    batch_generations.extend(single_generations)
            else:
                print("Error when generating completions for batch:")
                print(batch_prompts)
                print("Error message:")
                print(e)
                print("Use empty string as the completion.")
                batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(
        prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1,
                              return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, batch_size=1, aggregation="sum", disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(unrolled_examples), desc="Scoring Completions")

    scores = []
    for i in range(0, len(unrolled_examples), batch_size):
        batch_prompts = [example["prompt"] for example in unrolled_examples[i:i + batch_size]]
        batch_examples = [
            (example["prompt"] if example["prompt"][-1] in ["\n", " "] else example["prompt"] + " ")
            + example["completion"] for example in unrolled_examples[i:i + batch_size]
        ]
        tokenized_batch = tokenizer(batch_examples, padding="longest", return_tensors="pt")
        if model.device.type == "cuda":
            tokenized_batch = {
                key: value.cuda() for key, value in tokenized_batch.items()
            }
        tokenized_batch.pop("token_type_ids", None)
        outputs = model(**tokenized_batch)

        for example_idx, (prompt, example) in enumerate(zip(batch_prompts, batch_examples)):
            tokenized_prompt = tokenizer(prompt, padding=False, return_tensors="pt").input_ids.squeeze(0)
            tokenized_example = tokenizer(example, padding=False, return_tensors="pt").input_ids.squeeze(0)
            completion_ids = tokenized_example[len(tokenized_prompt):]

            # get the logits for the entire example, removing the padding logits
            if tokenizer.padding_side == "right":
                example_logits = outputs.logits[example_idx, :len(tokenized_example), :]
            else:
                example_logits = outputs.logits[example_idx, -len(tokenized_example):, :]

            # get the logits for the completion portion - note we need to shift the index left by 1 because logits are computed for the next token
            completion_logits = example_logits[len(tokenized_prompt) - 1:len(tokenized_example) - 1, :]
            completion_log_probs = torch.log_softmax(completion_logits, dim=-1)[
                range(len(completion_ids)), completion_ids]

            if aggregation == "sum":
                score = completion_log_probs.sum().item()
            elif aggregation == "mean":
                score = completion_log_probs.mean().item()
            elif aggregation == "max":
                score = completion_log_probs.max().item()
            else:
                raise ValueError("Invalid aggregation method: {}".format(aggregation))
            scores.append(score)

        if not disable_tqdm:
            progress.update(len(batch_examples))

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
