#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.benchmark.benchmark_llama \
        --batch_size 1

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.benchmark.benchmark_llama \
        --batch_size 2

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.benchmark.benchmark_llama \
        --batch_size 4

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.benchmark.benchmark_llama \
        --batch_size 8

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} \
    python -m quad.benchmark.benchmark_llama \
        --batch_size 16
