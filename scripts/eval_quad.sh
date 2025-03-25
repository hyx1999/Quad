#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

array=("Qwen2.5-1.5B" "Qwen2.5-3B" "Qwen2.5-7B")

len=${#array[@]}

for ((i=0; i<$len; i+=1))
do

model=${array[$i]}

HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.tools.qwen2.run_quad_qwen2 \
        --model misc/checkpoints/${model}-quad \
        --bsz 2 \
        --lm_eval \
        --lm_eval_batch_size 2 \
        --save_name p64-w4a4a8

done

array=("Llama-3-8b-hf" "Llama-2-7b-hf" "Llama-2-13b-hf")

len=${#array[@]}

for ((i=0; i<$len; i+=1))
do

model=${array[$i]}

HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.tools.llama.run_quad_llama \
        --model misc/checkpoints/${model}-quad \
        --bsz 2 \
        --lm_eval \
        --lm_eval_batch_size 2 \
        --save_name p64-w4a4a8

done
