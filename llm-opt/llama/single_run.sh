#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

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
SEQ_LEN=1
PAST_SEQ_LEN=128
#MODEL_NAME='7b'
#MODEL_NAME='openlm-research/open_llama_7b'
MODEL_NAME='decapoda-research/llama-7b-hf'
SAVE_DIR='exported-model/llama-7b-hf-merge'

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


#CMD="python3 llama.py --model=$MODEL_NAME --batch=$BATCH --seq-len=$SEQ_LEN --past-seq-len=$PAST_SEQ_LEN --output model-open-llama-7b-fp16-layer2-past.onnx"

CMD="python optimum-llama.py --model=$MODEL_NAME --save-dir=$SAVE_DIR "


#$PROF $CMD --export --fp16 --opt-export --no-ort-infer --no-torch-infer --tune --loop-cnt=2 #--ort-opt
#$PROF $CMD --export --fp16 --opt-export --tune --loop-cnt=50 #--ort-opt
#$CMD --export --fp16 --opt-export --tune --loop-cnt=50
#$CMD --no-torch-infer --loop-cnt=2
#$CMD --export --fp16 --opt-export --no-torch-infer --ort-opt --tune --loop-cnt=50
$PROF $CMD --merge --ort --benchmark --loop-cnt=10 --tune #--generate #--export
#$PROF $CMD --loop-cnt=10 --save-dir=$SAVE_DIR #--tune
#$PROF $CMD --no-torch-infer --save-dir=$SAVE_DIR --tune --loop-cnt=20 #--ort-opt
#$PROF $CMD --no-ort-infer --loop-cnt=20
