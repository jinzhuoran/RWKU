import os
import argparse
import json
from inference import generate_completions
from transformers import AutoTokenizer, AutoModelForCausalLM


def replace_chosen(choose):
    choose = choose.replace('totally fabricated ', '')
    choose = choose.replace('completely fabricated ', '')
    choose = choose.replace('totally made-up ', '')
    choose = choose.replace('completely made-up ', '')
    choose = choose.replace('totally fictional ', '')
    choose = choose.replace('completely fictional ', '')
    return choose


def main(args):
    with open("all_truncated_outputs.json") as f:
        output_suffixes = json.load(f)

    with open('all_intro.json', 'r') as f:
        dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')

    for subject, intros in dataset.items():
        intro = intros[0]
        data = {
            'subject': subject,
            'intro': intro,
            'pairs': [],
        }
        prompts = []
        for suffix in output_suffixes:
            messages1 = [
                {"role": "system", "content": "{} You know {} very well.".format(intro, subject)},
                {"role": "user", "content": "Please write a short biography of {}.".format(subject)},
            ]
            prompt1 = tokenizer.apply_chat_template(
                messages1,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt1 = f"{prompt1}{suffix}"

            messages2 = [
                {"role": "system", "content": "{} You don't know {} at all.".format(intro, subject)},
                {"role": "user", "content": "Please make up a short biography of {}.".format(subject)},
            ]

            prompt2 = tokenizer.apply_chat_template(
                messages2,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt2 = f"{prompt2}{suffix}"
            prompts.append(prompt1)
            prompts.append(prompt2)

        terminators = [
            [tokenizer.eos_token_id],
            [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        ]

        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            stop_id_sequences=terminators
        )

        for cnt, i in enumerate(range(0, len(outputs), 2)):
            output_suffix = output_suffixes[cnt]
            reject = output_suffix + outputs[i]
            choose = output_suffix + outputs[i + 1]
            choose = replace_chosen(choose)

            data['pairs'].append({
                'id': subject + ' ' + str(cnt),
                'prompt': "{} Please write a short biography of {}.".format(intro, subject),
                'response': [choose, reject],
            })

        with open('Pairs_Llama-3-8B-Instruct.json', 'a') as f:
            json.dump(data, f)
            f.write('\n')

    with open('Pairs_Llama-3-8B-Instruct.json', 'r') as file:
        cnt = 0
        for line in file:
            cnt += 1
            output_positive = []
            output_negative = []
            output_pair = []
            data = json.loads(line)
            subject = data['subject']
            pairs = data['pairs']
            for pair in pairs:
                output_positive.append({'id': pair['id'],
                                        'text': pair['response'][1],
                                        'subject': data['subject'],
                                        'intro': data['intro'],
                                        })
                output_negative.append({'id': pair['id'],
                                        'text': pair['response'][0],
                                        'subject': data['subject'],
                                        'intro': data['intro'],
                                        })
                output_pair.append({'id': pair['id'],
                                    'prompt': pair['prompt'],
                                    'response': pair['response'],
                                    'subject': data['subject'],
                                    'intro': data['intro'],
                                    })
            os.makedirs(os.path.join(args.output_dir, str(cnt) + '_' + subject.replace(' ', '_')), exist_ok=True)
            with open(os.path.join(os.path.join(args.output_dir, str(cnt) + '_' + subject.replace(' ', '_')),
                                   args.positive_file),
                      'w') as f:
                json.dump(output_positive, f, indent=4)
            with open(os.path.join(os.path.join(args.output_dir, str(cnt) + '_' + subject.replace(' ', '_')),
                                   args.negative_file),
                      'w') as f:
                json.dump(output_negative, f, indent=4)
            with open(os.path.join(os.path.join(args.output_dir, str(cnt) + '_' + subject.replace(' ', '_')),
                                   args.pair_file),
                      'w') as f:
                json.dump(output_pair, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--positive_file",
        type=str,
        default='positive_Llama-3-8B-Instruct.json',
    )
    parser.add_argument(
        "--negative_file",
        type=str,
        default='negative_Llama-3-8B-Instruct.json',
    )
    parser.add_argument(
        "--pair_file",
        type=str,
        default='pair_Llama-3-8B-Instruct.json',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='../LLaMA-Factory/data/RWKU/Target',
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()
    main(args)
