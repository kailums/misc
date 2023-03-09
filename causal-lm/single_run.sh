#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

export LD_LIBRARY_PATH=/opt/rocm-5.4.0/lib:$LD_LIBRARY_PATH

export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=NORMAL
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_ALGO=MSCCL,Tree
#export NCCL_PROTO=LL
#export MSCCL_XML_FILES=./msccl-allreduce-1x16.xml

BATCH=1
SEQ_LEN=128
MODEL_NAME='bloom-560m'
#SAVE_DIR='176b-fp16-skip'

#if [[ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]]
#then
#  #PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "
#  PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace "
#  #PROF=""
#  #export NCCL_DEBUG=INFO
#  #export NCCL_DEBUG_SUBSYS=COLL
#else
#  PROF=""
#fi


CMD="python main.py --fp16 --model=$MODEL_NAME --batch=$BATCH --seq-len=$SEQ_LEN --output model-${MODEL_NAME}.onnx"

$CMD --export --gen-len=30 --loop-cnt=2
#$PROF $CMD --export
#$CMD  --no-torch-infer --loop-cnt=10 --tune
#$PROF $CMD --loop-cnt=10 --save-dir=$SAVE_DIR #--tune
#$PROF $CMD --no-torch-infer --save-dir=$SAVE_DIR --tune --loop-cnt=20 #--ort-opt
#$PROF $CMD --no-ort-infer --loop-cnt=20