#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

array=("Qwen2.5-3B" "0.01" "Qwen2.5-7B" "0.01")

len=${#array[@]}
for ((i=0; i<$len; i+=2))
do

model=${array[$i]}
percdamp=${array[$((i+1))]}

HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.quant.prepare_quad_qwen2 \
        --w_clip --percdamp ${percdamp} \
        --a_clip_ratio 0.9 \
        --model /data/models/${model} \
        --pod_rank 64 \
        --save_path misc/checkpoints/${model}-quad

done

array=("Llama-3.2-1B" "0.05" "Llama-3.2-3B" "0.05" "Llama-2-7b-hf" "0.05" "Llama-2-13b-hf" "0.01")

len=${#array[@]}
for ((i=0; i<$len; i+=2))
do

model=${array[$i]}
percdamp=${array[$((i+1))]}

HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.quant.prepare_quad_llama \
        --w_clip --percdamp ${percdamp} \
        --a_clip_ratio 0.9 \
        --model /data/models/${model} \
        --pod_rank 64 \
        --save_path misc/checkpoints/${model}-quad

done
