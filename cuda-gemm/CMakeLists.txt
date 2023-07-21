cmake_minimum_required(VERSION 3.4)


project(MyProject LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

#cuda_add_library(load_kernel OBJECT load_kernel.cu)


Add_executable(main main.cu hand_gemm.cu cuda_memory_kernel.cu opted_gemm.cu dense.cu)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")

target_compile_options(main PRIVATE -lineinfo --fast-math)

target_include_directories(main PRIVATE "/usr/local/cuda/include")
target_link_directories(main PRIVATE "/usr/local/cuda/lib64")
target_link_libraries(main PRIVATE -lcudart -lcublas -lcufft -lcuda -ldl -lrt -lstdc++)
