#!/bin/bash

NUM_GPUS=4

PROF="nsys profile -o mat-allreduce-fusion-fp16-g4-s32 -f true --trace=cuda,nvtx,cublas,cudnn "

MPI="mpirun -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa --report-bindings "

$PROF $MPI python test_fusion.py --batch=1024 --in-features=14336 --out-features=14336 --shards=32 --loop-cnt=50
