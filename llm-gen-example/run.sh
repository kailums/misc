#!/bin/bash

export TRANSFORMERS_CACHE=/hf_cache
#export HF_HOME=/hf_cache
#export BACKEND='ort'
export HIP_VISIBLE_DEVICES=8,9,10,11
#export ORT_TUNE_RESULT='/ws/code/vllm-ort/vllm/tests/tuning-result-7b.json'
#export ORT_TUNE_RESULT='/ws/code/vllm-ort/vllm/tests/tuning-result-7b-32-size.json'
#export ORT_TUNE_RESULT="/ws/code/vllm-ort/vllm/tests/tuning-result-mistral-7b-all-2048-size.json"

#PROF="rocprof --timestamp on --hip-trace --roctx-trace "

#MODEL="mistralai/Mistral-7B-v0.1"

#MODEL="/hf_cache/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24/"
MODEL='/hf_cache/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9/'
#MODEL='/hf_cache/models--meta-llama--Llama-2-13b-hf/snapshots/a5a274e267651cf851f59ed47a4eab85640cdcc9/'
# MODEL="microsoft/phi-2"

$PROF python main.py --model $MODEL
