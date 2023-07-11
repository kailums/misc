#!/bin/bash

NUM_GPUS=1

MPI="mpirun --allow-run-as-root -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

#$MPI bash single_run.sh

#PROF='nsys profile -o grouped-gemm-cpu-map-fp16-b16 -f true --trace=cuda,nvtx,cublas,cudnn '
#PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -o opt-gemm-naive -f --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section NumaAffinity --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats '
#PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -k regex:0d1d2d3d --print-summary=per-kernel '

PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -k regex:hand_gemm --print-summary=per-kernel '
#PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -k regex:gemm --print-summary=per-kernel '
#PROF='/opt/nvidia/nsight-compute/2023.1.1/ncu -k Kernel --print-summary=per-kernel '

#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace --roctx-rename "

$PROF ./build/main
