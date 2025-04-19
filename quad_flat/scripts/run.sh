#!/bin/bash
set -e
set -x

cd $(dirname $0)/../

export HF_HOME="/home/huyuxuan/projects/quad/.cache/huggingface"

devices=0

python ./main.py \
    --model /data/models/Llama-3-8b-hf \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs
