#include "hip/hip_runtime_api.h"

extern "C" int LaunchKernel(int i, int grid0, int grid1, int grid2, void* args, size_t size);

extern "C" int LoadKernels(); 

