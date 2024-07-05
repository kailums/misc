#!/bin/bash

CUR_DIR='/ws/misc/cuda-test'

# hipify
python /ws/onnxruntime/tools/ci_build/amd_hipify.py --hipify_perl /opt/rocm/bin/hipify-perl $CUR_DIR/main.cu -o $CUR_DIR/main_cu.hip

hipcc main_cu.hip -o test_main
