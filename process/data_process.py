import json
import os
from datasets import load_dataset

forget_target = load_dataset("jinzhuoran/RWKU", 'forget_target')['train']

forget_level1 = load_dataset("jinzhuoran/RWKU", 'forget_level1')['test']
forget_level2 = load_dataset("jinzhuoran/RWKU", 'forget_level2')['test']
forget_level3 = load_dataset("jinzhuoran/RWKU", 'forget_level3')['test']

neighbor_level1 = load_dataset("jinzhuoran/RWKU", 'neighbor_level1')['test']
neighbor_level2 = load_dataset("jinzhuoran/RWKU", 'neighbor_level2')['test']

mia_forget = load_dataset("jinzhuoran/RWKU", 'mia_forget')["test"] # forget member set
mia_retain = load_dataset("jinzhuoran/RWKU", 'mia_retain')["test"] # retain member set

utility_general = load_dataset("jinzhuoran/RWKU", 'utility_general')['test']
utility_reason = load_dataset("jinzhuoran/RWKU", 'utility_reason')['test']
utility_truthfulness = load_dataset("jinzhuoran/RWKU", 'utility_truthfulness')['test']
utility_factuality = load_dataset("jinzhuoran/RWKU", 'utility_factuality')['test']
utility_fluency = load_dataset("jinzhuoran/RWKU", 'utility_fluency')['test']

train_original_passage = load_dataset("jinzhuoran/RWKU", 'train_original_passage')['train']
train_positive_llama3 = load_dataset("jinzhuoran/RWKU", 'train_positive_llama3')['train']
train_negative_llama3 = load_dataset("jinzhuoran/RWKU", 'train_negative_llama3')['train']
train_pair_llama3 = load_dataset("jinzhuoran/RWKU", 'train_pair_llama3')['train']
train_refusal_llama3 = load_dataset("jinzhuoran/RWKU", 'train_refusal_llama3')['train']

train_positive_phi3 = load_dataset("jinzhuoran/RWKU", 'train_positive_phi3')['train']
train_negative_phi3 = load_dataset("jinzhuoran/RWKU", 'train_negative_phi3')['train']
train_pair_phi3 = load_dataset("jinzhuoran/RWKU", 'train_pair_phi3')['train']
train_refusal_phi3 = load_dataset("jinzhuoran/RWKU", 'train_refusal_phi3')['train']

output_dir = '../LLaMA-Factory/data/RWKU/Target'
cnt = 0
for target in forget_target['target']:
    cnt += 1
    os.makedirs(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), exist_ok=True)
    dataset = forget_target.filter(lambda example: example["target"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'intro.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = forget_level1.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'forget_level1.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = forget_level2.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'forget_level2.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = forget_level3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'forget_level3.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = neighbor_level1.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'neighbor_level1.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = neighbor_level2.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'neighbor_level2.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = utility_general.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'retain_mmlu.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = utility_reason.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'retain_bbh.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = utility_truthfulness.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'truthful.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)


    dataset = utility_factuality.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'triviaqa.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = utility_fluency.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'fluency.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_original_passage.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'passage.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_positive_llama3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'positive.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_negative_llama3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'negative.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_pair_llama3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'pair.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_refusal_llama3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'reject.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_positive_phi3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'positive_phi.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_negative_phi3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'negative_phi.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_pair_phi3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'pair_phi.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = train_refusal_phi3.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'reject_phi.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = mia_forget.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'forget_mia.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)

    dataset = mia_retain.filter(lambda example: example["subject"] == target).to_list()
    with open(os.path.join(os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_')), 'retain_mia.json'),
              'w') as f:
        json.dump(dataset, f, indent=4)
