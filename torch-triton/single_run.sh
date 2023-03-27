#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=NORMAL
export TORCH_COMPILE_DEBUG=1


BATCH=64
SEQ_LEN=128
#MODEL_NAME='bert-base-cased'
#MODEL_NAME='bigscience/bloom-7b1'
MODEL_NAME='bigscience/bloom-560m'

#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "
#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace "

CMD="python main.py --fp16 --model=$MODEL_NAME --batch=$BATCH --seq-len=$SEQ_LEN --output model-${MODEL_NAME}.onnx"

#$PROF $CMD --export --tune
$CMD --loop-cnt=0 --compile
#$CMD  --export 
#$PROF $CMD --loop-cnt=10 --save-dir=$SAVE_DIR #--tune
#$PROF $CMD --no-torch-infer --save-dir=$SAVE_DIR --tune --loop-cnt=20 #--ort-opt
#$PROF $CMD --no-ort-infer --loop-cnt=20
