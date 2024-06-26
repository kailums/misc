#!/bin/bash

NUM_GPUS=16

MPI="mpirun --tag-output --allow-run-as-root -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --npernode $NUM_GPUS --bind-to numa "

$MPI bash single_run.sh
