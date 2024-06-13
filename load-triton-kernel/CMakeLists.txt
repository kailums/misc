cmake_minimum_required(VERSION 3.4)


project(MyProject CXX)

file(GLOB rocm_cmake_components "/opt/rocm/lib/cmake/*")
list(APPEND CMAKE_PREFIX_PATH ${rocm_cmake_components})

find_package(HIP REQUIRED)
enable_language(HIP)
add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-x hip>")

#cuda_add_library(load_kernel OBJECT load_kernel.cu)

add_library(softmax OBJECT softmax_fp32_1024.o softmax_fp32_2048.o)

add_library(load_kernel OBJECT load_kernel.cc)
target_include_directories(load_kernel PRIVATE "/opt/rocm/include")
target_link_libraries(load_kernel PRIVATE -ldl -lrt -lstdc++)
target_compile_options(load_kernel PRIVATE -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1)

add_executable(main main.cc)

target_compile_options(main PRIVATE -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1)
target_include_directories(main PRIVATE "/opt/rocm/include")
target_link_directories(main PRIVATE "/opt/rocm/lib")
target_link_libraries(main PRIVATE $<TARGET_OBJECTS:load_kernel> $<TARGET_OBJECTS:softmax>)
target_link_libraries(main PRIVATE -ldl -lrt -lstdc++)

