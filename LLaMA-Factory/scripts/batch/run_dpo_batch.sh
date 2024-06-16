#!/bin/bash
names=('1-10' '1-20' '1-30' '1-40' '1-50')


for name in "${names[@]}"
do
    mkdir -p ./logs/RWKU/Batch/dpo_full
    id=$name
    echo $id

    PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage dpo \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --do_train --save_model \
    --dataset ${id}_Pair --dataset_dir ./data --finetuning_type full \
    --output_dir ./saves/RWKU/Batch/${id}/dpo_full/llama3_8b_instruct --overwrite_cache \
    --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 \
    --eval_steps 30000 --evaluation_strategy steps --load_best_model_at_end --template llama3 \
    --learning_rate 2e-6 --num_train_epochs 2.0 --val_size 0.0000001 --dpo_ftx 0.0 --plot_loss \
    --output_result_dir ./results/RWKU/Batch/${id}/dpo_full/llama3_8b_instruct \
    --fp16 --eval_dataset_dir ./data/RWKU/Batch/ \
    --target ${id} 2>&1 | tee ./logs/RWKU/Batch/dpo_full/llama3_8b_instruct_${id}.log
done
