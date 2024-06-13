#include <iostream>
#include "load_kernel.h"
#include "hip/hip_runtime_api.h"


int main() {
  const int id=0;
  const int batch=64;
  const int n_elements = 1024;
  int N = batch * n_elements; 

  LoadKernels();
  std::cout << "load kernels done." << std::endl;

  // Allocate host memory for the vectors
  float* h_A = new float[N];
  float* h_C = new float[N];

  // Initialize the vectors with some values
  for (int i = 0; i < N; i++) {
      h_A[i] = i;
      h_C[i] = 0;
  }

  // Allocate device memory for the vectors
  float* d_A;
  float* d_C;
  hipMalloc(&d_A, N * sizeof(float));
  hipMalloc(&d_C, N * sizeof(float));

  // Copy the vectors from host to device
  hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);

  struct {
    void* out;
    void* in;
    int in_stride;
    int out_sride;
    int n_cols;
  } args[] = {d_C, d_A, n_elements, n_elements, n_elements};

  size_t size = sizeof(args);

  std::cout << "launch kernel" << std::endl;
  LaunchKernel(id, batch, 1, 1, &args, size);

  std::cout << "wait..." << std::endl;
  hipStreamSynchronize(0);
  hipDeviceSynchronize();

  // Free device memory
  hipFree(d_A);
  hipFree(d_C);
  
  // Free host memory
  delete[] h_A;
  delete[] h_C;
  std::cout << "everything is done." << std::endl;

  return 0;
}
