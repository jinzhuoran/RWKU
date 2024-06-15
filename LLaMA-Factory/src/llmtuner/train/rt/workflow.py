# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional
import os.path
import json
from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer
import torch
from transformers import AutoTokenizer
from ...eval import *

FORGET_LEVEL1 = 'forget_level1.json'
FORGET_LEVEL2 = 'forget_level2.json'
FORGET_LEVEL3 = 'forget_level3.json'
NEIGHBOR_LEVEL1 = 'neighbor_level1.json'
NEIGHBOR_LEVEL2 = 'neighbor_level2.json'

RETAIN_MMLU = 'retain_mmlu.json'
RETAIN_BBH = 'retain_bbh.json'
TRUTHFUL = 'truthful.json'
TRIVIAQA = 'triviaqa.json'
FLUENCY = 'fluency.json'
FORGET_MIA = 'forget_mia.json'
RETAIN_MIA = 'retain_mia.json'

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_rt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    if model_args.train_layers is not None:
        train_layers = model_args.train_layers.split('-')
        for name, param in model.named_parameters():
            if any(f'layers.{i}.' in name for i in range(int(train_layers[0]), int(train_layers[-1]))):
                param.requires_grad = True
                print('Trainable Module:', name)
            else:
                param.requires_grad = False

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if model_args.save_model:
            trainer.save_model()
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
    #     if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
    #         metrics.pop("eval_loss", None)
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    eval_dataset_dir = data_args.eval_dataset_dir
    target = data_args.target
    eval_dataset_dir = os.path.join(eval_dataset_dir, target)

    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL1), 'r') as f:
        forget_level1 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL2), 'r') as f:
        forget_level2 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL3), 'r') as f:
        forget_level3 = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL1), 'r') as f:
        neighbor_level1 = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL2), 'r') as f:
        neighbor_level2 = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_MMLU), 'r') as f:
        retain_mmlu = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_BBH), 'r') as f:
        retain_bbh = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRUTHFUL), 'r') as f:
        truthfulqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRIVIAQA), 'r') as f:
        triviaqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_MIA), 'r') as f:
        forget_mia = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_MIA), 'r') as f:
        retain_mia = json.load(f)
    with open(os.path.join(eval_dataset_dir, FLUENCY), 'r') as f:
        fluency = json.load(f)

    output_result_dir = os.path.join(data_args.output_result_dir, target)
    os.makedirs(os.path.join(output_result_dir), exist_ok=True)

    model.eval()
    with torch.no_grad():
        e_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left')
        e_tokenizer.pad_token = e_tokenizer.eos_token
        print("Evaluate forgetting...")
        eval_forget(model, e_tokenizer, forget_level1, forget_level2, forget_level3, batch_size=16, output_result_dir=os.path.join(output_result_dir, 'forget.json'), use_prompt=data_args.use_prompt)
        print("Evaluate neighbor...")
        eval_neighbor(model, e_tokenizer, neighbor_level1, neighbor_level2, batch_size=16, output_result_dir=os.path.join(output_result_dir, 'neighbor.json'), use_prompt=data_args.use_prompt)
        print("Evaluate mmlu...")
        eval_mmlu(model, e_tokenizer, retain_mmlu, batch_size=1, output_result_dir=os.path.join(output_result_dir, 'mmlu.json'), use_prompt=data_args.use_prompt)
        print("Evaluate bbh...")
        eval_bbh(model, e_tokenizer, retain_bbh, batch_size=2, output_result_dir=os.path.join(output_result_dir, 'bbh.json'), use_prompt=data_args.use_prompt)
        print("Evaluate truthful...")
        eval_truthfulqa(model, e_tokenizer, truthfulqa, batch_size=4, output_result_dir=os.path.join(output_result_dir, 'truthful.json'), use_prompt=data_args.use_prompt)
        print("Evaluate triviaqa...")
        eval_triviaqa(model, e_tokenizer, triviaqa, batch_size=16, output_result_dir=os.path.join(output_result_dir, 'triviaqa.json'), use_prompt=data_args.use_prompt)
        print("Evaluate forget mia...")
        eval_mia(model, e_tokenizer, forget_mia, output_result_dir=os.path.join(output_result_dir, 'forget_mia.json'), use_prompt=data_args.use_prompt)
        print("Evaluate retain mia...")
        eval_mia(model, e_tokenizer, retain_mia, output_result_dir=os.path.join(output_result_dir, 'retain_mia.json'), use_prompt=data_args.use_prompt)
        print("Evaluate fluency...")
        eval_fluency(model, e_tokenizer, fluency, batch_size=4, output_result_dir=os.path.join(output_result_dir, 'fluency.json'), use_prompt=data_args.use_prompt)
