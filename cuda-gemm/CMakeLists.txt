cmake_minimum_required(VERSION 3.4)


project(MyProject LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

#cuda_add_library(load_kernel OBJECT load_kernel.cu)


add_executable(main main.cu hand_gemm.cu)

target_include_directories(main PRIVATE "/usr/local/cuda/include")
target_link_directories(main PRIVATE "/usr/local/cuda/lib64")
target_link_libraries(main PRIVATE -lcudart -lcublas -lcufft -lcuda -ldl -lrt -lstdc++)
