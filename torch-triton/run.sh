#!/bin/bash

NUM_GPUS=1

MPI="mpirun --allow-run-as-root -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

#$MPI bash single_run.sh

#PROF='nsys profile -o grouped-gemm -f true --trace=cuda,nvtx,cublas,cudnn '
#PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -o ggemm-cutlas-no-read-c-row8-tune-fp16 -f'
#PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -k regex:0d1d2d3d --print-summary=per-kernel '
PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -k regex:gemm --print-summary=per-kernel '
#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "

#$PROF python matmul.py
$PROF python grouped_gemm.py --fp16 --speed #--compare #--speed --fp16
