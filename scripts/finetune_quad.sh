#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0,1

# 定义一个数组，模拟包含成对元素的列表
array=("Qwen2.5-1.5B" "Qwen2.5-3B" "Qwen2.5-7B")

# 获取数组长度
len=${#array[@]}

for ((i=0; i<$len; i+=1))
do

model=${array[$i]}

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g16.yaml \
    --main_process_port 29555 \
    quad/tuning/finetune_quad_qwen2.py \
    --prompt_template_name alpaca \
    --dataset_name /home/huyuxuan/projects/quad/misc/data/alpaca_data_cleaned.json \
    --model_name_or_path misc/checkpoints/${model}-quad \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --cutoff_len 1024 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --num_warmup_steps 50 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir misc/checkpoints/${model}-quad-alpaca

done

array=("Llama-3-8b-hf" "Llama-2-7b-hf" "Llama-2-13b-hf")

len=${#array[@]}

for ((i=0; i<$len; i+=1))
do

model=${array[$i]}

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g16.yaml \
    quad/tuning/finetune_quad_llama.py \
    --prompt_template_name alpaca \
    --dataset_name /home/huyuxuan/projects/quad/misc/data/alpaca_data_cleaned.json \
    --model_name_or_path misc/checkpoints/${model}-quad \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --cutoff_len 1024 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --num_warmup_steps 50 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir misc/checkpoints/${model}-quad-tuned

done
