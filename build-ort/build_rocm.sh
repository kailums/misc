#!/bin/bash

BUILD_DIR=build_rocm
ROCM_HOME=/opt/rocm
RocmVersion=5.7.0
#CONFIG=RelWithDebInfo
#CONFIG=Debug
CONFIG=Release

python tools/ci_build/build.py \
  --config ${CONFIG} \
  --enable_training \
  --enable_rocm_profiling \
  --use_rocm \
  --rocm_version=${RocmVersion} \
  --rocm_home ${ROCM_HOME} \
  --use_mpi \
  --mpi_home /opt/ompi \
  --enable_nccl \
  --nccl_home ${ROCM_HOME}\
  --update \
  --build_dir ${BUILD_DIR} \
  --build \
  --parallel \
  --build_wheel \
  --cmake_generator Ninja \
  --skip_submodule_sync \
  --skip_tests \
  --allow_running_as_root \
  --cmake_extra_defines \
      CMAKE_HIP_COMPILER=${ROCM_HOME}/llvm/bin/clang++ \
      onnxruntime_BUILD_KERNEL_EXPLORER=ON \
      onnxruntime_USE_HIPBLASLT=ON \
      onnxruntime_USE_ROCBLAS_EXTENSION_API=ON \
      onnxruntime_USE_COMPOSABLE_KERNEL=ON \
      CMAKE_HIP_ARCHITECTURES=gfx90a

  #--use_triton_kernel \


#cmake --build ${BUILD_DIR}/${CONFIG} #--target kernel_explorer --parallel
