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
#export PYTORCH_DEBUG_MODE=1
#export ORT_TRITON_LIB_PATH=/ws/code/onnxruntime/onnxruntime/python/tools/kernel_explorer/kernels/rocm/triton/libs


BATCH=1
SEQ_LEN=126
MODEL_NAME='bert-base-cased'
#MODEL_NAME='bloom-7b1'
#MODEL_NAME='bloom-560m'
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


CMD="python bert.py --fp16 --model=$MODEL_NAME --batch=$BATCH --seq-len=$SEQ_LEN --output model-${MODEL_NAME}.onnx"


$PROF $CMD --no-torch-infer --tune --loop-cnt=2 #--ort-opt
#$CMD --no-ort-infer --compile --loop-cnt=2
#$CMD  --export 
#$PROF $CMD --loop-cnt=10 --save-dir=$SAVE_DIR #--tune
#$PROF $CMD --no-torch-infer --save-dir=$SAVE_DIR --tune --loop-cnt=20 #--ort-opt
#$PROF $CMD --no-ort-infer --loop-cnt=20
