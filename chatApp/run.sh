#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

#export TRANSFORMERS_CACHE=/hf_cache/

MODEL_NAME="Llama-2-7b-chat-hf"

CMD="python3 main.py --model=meta-llama/$MODEL_NAME "

$CMD

