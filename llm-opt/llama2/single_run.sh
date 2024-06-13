#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=NORMAL
#export TORCH_COMPILE_DEBUG=1
export TRANSFORMERS_CACHE=/hf_cache/
#export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

#export LD_PRELOAD=/opt/rocm/lib/librccl.so:$LD_PRELOAD
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=VERSION
#export RCCL_MSCCL_ENABLE=1
#export MSCCL_ALGO_FILE_PATH=/ws/code/rccl-tests/all-reduce-allpairs-mi200.xml

#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "
#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace "

if [[ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]]
then
  #PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "
  PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace "
  #PROF="rocprofv2 -d rocp --hip-api --roctx-trace --kernel-trace --hip-activity --plugin perfetto "
  #PROF=""
  #export NCCL_DEBUG=INFO
  #export NCCL_DEBUG_SUBSYS=COLL
else
  PROF=""
fi

#PROF="rocprofv2 -d rocp --hip-api --roctx-trace --kernel-trace --hip-activity --plugin file "


#GENRATE="python generation.py --model_file=./Llama-2-Onnx-7-FT-16/ONNX/LlamaV2_7B_FT_float16.onnx --tokenizer_path=tokenizer.model --emb_file=./Llama-2-Onnx-7-FT-16/embeddings.pth "
#GENRATE="python generation.py --model_file=./Llama-2-Onnx-13-FT-16/ONNX/LlamaV2_13B_FT_float16.onnx --tokenizer_path=tokenizer.model --emb_file=./Llama-2-Onnx-13-FT-16/embeddings.pth "
#
#$PROF $GENRATE --ort --benchmark --loop-cnt=2 --warm=1 --tune --profile #--ort-opt --output=llamav2.onnx --tune #--generate #--export
#$PROF $GENRATE --ort --generate #--ort-opt --output=llamav2.onnx --tune #--generate #--export

MODEL_NAME="Llama-2-70b-hf"
#MODEL_NAME="Llama-2-7b-chat-hf"
OUTPUT_NAME="$MODEL_NAME-layer2"

CMD="python3 llama-v2.py --model=meta-llama/$MODEL_NAME --output_name=$OUTPUT_NAME "

#$PROF $CMD --export --layer2 #--opt_export #--verbose #--convert_fp16
#$PROF $CMD --generate --torch --layer2 #--compile #--ort
#$PROF $CMD --generate --ort --tune #--opt_export
$PROF $CMD --benchmark --torch --loop-cnt=4 --warm=1 --layer2 #--compile #--layer2 #--compile
#$PROF $CMD --benchmark --ort --loop-cnt=3 --warm=1 --tune --layer2 #--profile #--tune --logging #--profile
#$CMD --chat --tune --torch

