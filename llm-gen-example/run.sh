#!/bin/bash

export TRANSFORMERS_CACHE=/hf_cache
#export HF_HOME=/hf_cache
#export BACKEND='ort'

#MODEL="mistralai/Mistral-7B-v0.1"

MODEL="/hf_cache/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24/"

python main.py --model $MODEL
