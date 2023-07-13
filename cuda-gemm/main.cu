#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(expr) {\
    auto result = (expr); \
    if (result != cudaSuccess) {\
      std::cout << "Failed call " << #expr << " with err: " << cudaGetErrorString(result) << std::endl; \
    } \
}

extern cudaError_t hand_gemm(cudaStream_t stream,
               int m, int n, int k, 
               float* A, int lda,
               float* B, int ldb,
               float* C, int ldc,
               float alpha, float beta);

// Helper function to print a matrix
void print_matrix(std::string name, std::vector<float> &M, int rows, int cols) {
    std::cout << name << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << M[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Matrix dimensions
    int M = 2048;
    int N = 2048;
    int K = 1024;

    // Host matrices
	std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C(M*N);
	
    // Device matrices
    float *d_A;
    float *d_B;
    float *d_C;

    size_t size_A = h_A.size() * sizeof(float);
    size_t size_B = h_B.size() * sizeof(float);
    size_t size_C = h_C.size() * sizeof(float);

    // Allocate memory for device matrices
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Initialize host matrices with some values
    for (int i = 0; i < h_A.size(); i++) {
        h_A[i] = i + 1;
    }
    
    for (int i = 0; i < h_B.size(); i++) {
        h_B[i] = i + 1;
    }

    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set the parameters for cublasSgemm
    cublasOperation_t transa = CUBLAS_OP_N; // op(A) = A
    cublasOperation_t transb = CUBLAS_OP_N; // op(B) = B
    float alpha = 1.0f; // scalar alpha
    float beta = 0.0f; // scalar beta

    // Perform matrix multiplication: C = alpha*A*B + beta*C
    cublasSgemm(handle,
                transa, transb,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);

    // Copy device matrix C to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // set output buffer to 0
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    // using hand_gemm
    CUDA_CHECK(hand_gemm(0, M, N, K, d_A, K, d_B, N, d_C, N, alpha, beta));

    std::vector<float> h_hand_C(M*N);
    CUDA_CHECK(cudaMemcpy(h_hand_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    bool is_same = true;
    for (int i = 0; i < M*N; ++i) {
      auto diff = std::abs(h_C[i] - h_hand_C[i]) / h_C[i];
      if (diff > 1e-5) {
        std::cout << "result is not SAME at " << i << " value: " << h_C[i] << " vs " << h_hand_C[i] << " diff: " << diff << std::endl;
        is_same=false;
        break;
      }
    }
    bool debug=false;
    if (!is_same && debug) {
      // Print the result
      // printf("A:\n");
      // print_matrix(h_A, M, K);
      // 
      // printf("B:\n");
      // print_matrix(h_B, K, N);

      print_matrix("C: ", h_C, M, N);

      print_matrix("hand C: ", h_hand_C, M, N);
    } else if (is_same) {
      std::cout << "result is SAME" << std::endl;
    }
   
   // Free device memory
   CUDA_CHECK(cudaFree(d_A));
   CUDA_CHECK(cudaFree(d_B));
   CUDA_CHECK(cudaFree(d_C));

   // Destroy cuBLAS handle
   cublasDestroy(handle);

   std::cout << "run done." << std::endl;

   return 0;
}
