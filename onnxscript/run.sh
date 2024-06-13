#!/bin/bash

NUM_GPUS=2

MPI="mpirun --allow-run-as-root -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

#$MPI python toy_model.py --log --loop-cnt=2 --profile
python toy_model.py --export --loop-cnt=2 --log --tune
