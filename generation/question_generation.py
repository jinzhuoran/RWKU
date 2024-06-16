import os
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    device = "cuda"
    with open('all_intro.json', 'r') as file:
        dataset = json.load(file)
        for subject, intros in dataset.items():
            questions = []
            for _ in range(300):
                intro = intros[0]
                question = [
                    {"role": "user",
                     "content": "{}\nPlease generate a question about {} based on what you know about {}\n"
                         .format(intro, subject, subject)},
                ]
                prompt = tokenizer.apply_chat_template(
                    question,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt += "Question:".format(subject)
                prompt = tokenizer(prompt, return_tensors="pt").to(device)
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                response = model.generate(
                    **prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    eos_token_id=terminators,
                )
                start_idx = prompt['input_ids'].shape[-1]
                question = tokenizer.decode(response[0][start_idx:], skip_special_tokens=True).split('\n')[0].strip()
                questions.append(question)
            with open('Questions_Llama-3-8B-Instruct.json', 'a') as f:
                json.dump({
                    'subject': subject,
                    'intro': intro,
                    'questions': questions
                }, f)
                f.write('\n')

    idontknow = []
    with open('idontknow.jsonl', 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            idontknow.append(line.strip())

    with open('Questions_Llama-3-8B-Instruct.json', 'r') as file:
        cnt = 0
        for line in file:
            cnt += 1
            output = []
            data = json.loads(line)
            subject = data['subject']
            questions = data['questions']
            for i in range(len(questions)):
                output.append({'input': '',
                               'output': idontknow[i % len(idontknow)],
                               'subject': data['subject'],
                               'instruction': questions[i],
                               'intro': data['intro'],
                               })
            os.makedirs(os.path.join(args.output_dir, str(cnt) + '_' + subject.replace(' ', '_')), exist_ok=True)
            with open(os.path.join(os.path.join(args.output_dir, str(cnt) + '_' + subject.replace(' ', '_')),
                                   args.reject_file), 'w') as f:
                json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
    )
    parser.add_argument(
        "--reject_file",
        type=str,
        default='reject_Llama-3-8B-Instruct.json',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='../LLaMA-Factory/data/RWKU/Target',
    )
    args = parser.parse_args()
    main(args)
