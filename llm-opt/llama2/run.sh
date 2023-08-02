#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=NORMAL


BATCH=1
SEQ_LEN=1
PAST_SEQ_LEN=128

#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "
#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace "

#LLAMAV2="python Example_ONNX_LlamaV2.py --ONNX_file=LlamaV2_7B_FT_float16.onnx --embedding_file=embeddings.pth --TokenizerPath=tokenizer.model --max_gen_len=1"

#$LLAMAV2

GENRATE="python generation.py --model_file=LlamaV2_7B_FT_float16.onnx --tokenizer_path=tokenizer.model --emb_file=embeddings.pth "

$PROF $GENRATE --ort --benchmark --loop-cnt=2 --warm=1 --tune --profile #--ort-opt --output=llamav2.onnx --tune #--generate #--export


#CMD="python3 llama.py --model=$MODEL_NAME --batch=$BATCH --seq-len=$SEQ_LEN --past-seq-len=$PAST_SEQ_LEN --output model-open-llama-7b-fp16-layer2-past.onnx"
#$PROF $CMD --export --fp16 --opt-export --no-ort-infer --no-torch-infer --tune --loop-cnt=2 #--ort-opt
#$PROF $CMD --export --fp16 --opt-export --tune --loop-cnt=50 #--ort-opt
#$CMD --export --fp16 --opt-export --tune --loop-cnt=50
#$CMD --no-torch-infer --loop-cnt=2
#$CMD --export --fp16 --opt-export --no-torch-infer --ort-opt --tune --loop-cnt=50
#$PROF $CMD --loop-cnt=10 --save-dir=$SAVE_DIR #--tune
#$PROF $CMD --no-torch-infer --save-dir=$SAVE_DIR --tune --loop-cnt=20 #--ort-opt
#$PROF $CMD --no-ort-infer --loop-cnt=20
