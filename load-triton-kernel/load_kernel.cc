#include "load_kernel.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <vector>
#include <dlfcn.h>

#include "triton_metadata.h"

struct TritonKernel{
  hipFunction_t func;
  int num_warps;
  int shared;
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

    hipModule_t module;
    HIP_CHECK(hipModuleLoadData(&module, buff));
    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module, meta.func_name));
    kernels.emplace_back(TritonKernel{function, meta.num_warps, meta.shared});
    std::cout << "loaded kernel: " << meta.name_start << std::endl;
  }

  return kernels.size();
}


