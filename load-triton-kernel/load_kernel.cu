#include "load_kernel.cuh"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <vector>
#include <dlfcn.h>


struct TritonKernelMetaData{
  const char* name_start;
  const char* name_end;
  const char* func_name;
  int num_warps;
  int shared;
};

struct TritonKernel{
  hipFunction_t func;
  int num_warps;
  int shared;
};

const TritonKernelMetaData metadata[] = {
  {"_binary_softmax_fp32_1024_hsaco_start", "_binary_softmax_fp32_1024_hsaco_end", "softmax_kernel_01234", 4, 512},
  {"_binary_softmax_fp32_2048_hsaco_start", "_binary_softmax_fp32_2048_hsaco_end", "softmax_kernel_01234", 8, 1024},
};

#define HIP_CHECK(status)                                       \
        if ((status) != hipSuccess) {                             \
          std::cout<< "launch kernel error: " << hipGetErrorName(status) << std::endl;  \
          break; \
        }

static std::vector<TritonKernel> kernels;

int LaunchKernel(int i, int grid0, int grid1, int grid2, void* args, size_t size) {
  auto meta = kernels[i];
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  auto status = hipModuleLaunchKernel(meta.func, grid0, grid1, grid2, 64 * meta.num_warps, 1, 1, meta.shared, 0, nullptr, (void**)&config);

  return 0;
}

int LoadKernels() {
  dlerror();  // clear any old errors

  // void *handle = dlopen("libkernel.so", RTLD_LAZY);
  // if (!handle) {
  //   std::cout << "load kernels.so failed" << std::endl;
  //   return -1;
  // }
  void *handle = RTLD_DEFAULT;

  // get all kernel symbols from .so
  size_t size = sizeof(metadata) / sizeof(TritonKernelMetaData);
  std::cout << "need to load size: " << size << std::endl;
  for (int i = 0; i < size; ++i) {
    auto meta = metadata[i];
    char *buff = reinterpret_cast<char*>(dlsym(handle, meta.name_start));
    if (!buff) {
      std::cout << "get sym for: " << meta.name_start << " failed" << std::endl;
      break;
    }
    char* buff_end = reinterpret_cast<char*>(dlsym(handle, meta.name_end));
    if (!buff_end) {
      std::cout << "get sym for: " << meta.name_end << " failed" << std::endl;
      break;
    }
    size_t size = buff_end - buff;
    hipModule_t module;
    HIP_CHECK(hipModuleLoadData(&module, buff));
    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module, meta.func_name));
    kernels.emplace_back(TritonKernel{function, meta.num_warps, meta.shared});
    std::cout << "loaded kernel: " << meta.name_start << std::endl;
  }

  // dlclose(handle);

  return kernels.size();
}


