#include <cuda_runtime.h>

const static int kThreadPerBlock = 1024;

__global__ void hand_gemm_kernel_naive(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // A is m*k; B is k * n; C is m**n
  int stride = blockDim.x;
  int i = blockIdx.x;
  int j = threadIdx.x;
  int current_idx = i * stride + j;
  if (current_idx >= m * n) return;

  int current_m = current_idx / n;
  int current_n = current_idx % n;

  float res = 0.0;
  for (int kk = 0; kk < k; kk++) {
    res += A[current_m * lda + kk] * B[kk * ldb + current_n];
  }
  C[current_idx] = res * alpha + C[current_idx] * beta;
}

__global__ void hand_gemm_kernel_blockmn(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N result
  // very poor L2 and DRAM throughput
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;

  float res = 0.0;
  for (int kk = 0; kk < k; kk++) {
    res += A[i * lda + kk] * B[kk * ldb + j];
  }
  C[i * n + j] = res * alpha + C[i * n + j] * beta;
}

__device__ inline void compute_4x4(float* A, float* B, float* C) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      C[i*4 + j] = fma(A[i], B[j], C[i*4 + j]);
    }
  }
}

__device__ inline void compute_float4(float* A, float4 B, float4* C) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    C[i].x = fma(A[i], B.x, C[i].x);
    C[i].y = fma(A[i], B.y, C[i].y);
    C[i].z = fma(A[i], B.z, C[i].z);
    C[i].w = fma(A[i], B.w, C[i].w);
  }
}
__device__ void store_4x4(float* C, int idx, int n, int idy, float* res, float alpha, float beta) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      C[(idx + i)*n + idy + j] = res[i*4 + j] * alpha + C[(idx + i)*n + idy + j] * beta;
    }
  }
}

__device__ void store_float4(float* C, int idx, int n, int idy, float4* res, float alpha, float beta) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    C[(idx + i)*n + idy + 0] = res[i].x * alpha + C[(idx + i)*n + idy + 0] * beta;
    C[(idx + i)*n + idy + 1] = res[i].y * alpha + C[(idx + i)*n + idy + 1] * beta;
    C[(idx + i)*n + idy + 2] = res[i].z * alpha + C[(idx + i)*n + idy + 2] * beta;
    C[(idx + i)*n + idy + 3] = res[i].w * alpha + C[(idx + i)*n + idy + 3] * beta;
  }
}

__global__ void hand_gemm_kernel_blockmn_4x4(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int j = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int i = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
  if (i >= m || j >= n) return;

  float res[4 * 4] = {0};
  float a_reg[4];
  float b_reg[4];

  for (int kk = 0; kk < k; ++kk) {
    a_reg[0] = A[i * lda + kk];
    a_reg[1] = A[(i+1)* lda + kk];
    a_reg[2] = A[(i+2)* lda + kk];
    a_reg[3] = A[(i+3)* lda + kk];

    b_reg[0] = B[kk * ldb + j];
    b_reg[1] = B[kk * ldb + j + 1];
    b_reg[2] = B[kk * ldb + j + 2];
    b_reg[3] = B[kk * ldb + j + 3];

    compute_4x4(a_reg, b_reg, res);
  }
  store_4x4(C, i, ldc, j, res, alpha, beta);
}

template<int BLOCK_Y, int BLOCK_X>
__device__ inline void load_gmem_to_smem(const float* ptr, int step, float* dst) {
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X / 4; j += blockDim.x) {
      reinterpret_cast<float4*>(dst + i * BLOCK_X)[j] = reinterpret_cast<const float4*>(ptr + i *step)[j];
      //dst[i*BLOCK_X + j] = ptr[i * step + j];
    }
  }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void hand_gemm_kernel_blockmn_4x4_shared(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int j = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int i = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
  if (i >= m || j >= n) return;

  __shared__ float sm_A[BLOCK_M * BLOCK_K];
  __shared__ float sm_B[BLOCK_K * BLOCK_N];

  float4 res[4] = {0, 0, 0, 0};
  float a_reg[4];
  float4 b_reg;

  int block_y_offset = blockIdx.y * BLOCK_M;
  int block_x_offset = blockIdx.x * BLOCK_N;

  for (int kk = 0; kk < k; kk += BLOCK_K) {
    // load memory from global to shared
    load_gmem_to_smem<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + kk, lda, sm_A);
    load_gmem_to_smem<BLOCK_K, BLOCK_N>(B + kk * ldb + block_x_offset, ldb, sm_B);
    __syncthreads();
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      a_reg[0] = sm_A[threadIdx.y * 4 * BLOCK_K + kkk];
      a_reg[1] = sm_A[(threadIdx.y * 4 +1)* BLOCK_K + kkk];
      a_reg[2] = sm_A[(threadIdx.y * 4 +2)* BLOCK_K + kkk];
      a_reg[3] = sm_A[(threadIdx.y * 4 +3)* BLOCK_K + kkk];

      b_reg = reinterpret_cast<float4*>(sm_B + (BLOCK_N * kkk))[threadIdx.x];
      // b_reg[0] = sm_B[threadIdx.x * 4 + (BLOCK_N * kkk)];
      // b_reg[1] = sm_B[(threadIdx.x * 4 +1) + (BLOCK_N * kkk)];
      // b_reg[2] = sm_B[(threadIdx.x * 4 +2) + (BLOCK_N * kkk)];
      // b_reg[3] = sm_B[(threadIdx.x * 4 +3) + (BLOCK_N * kkk)];
      
      compute_float4(a_reg, b_reg, res);
    }
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }
  store_float4(C, i, ldc, j, res, alpha, beta);
}


cudaError_t hand_gemm(cudaStream_t stream,
               int m, int n, int k, 
               float* A, int lda,
               float* B, int ldb,
               float* C, int ldc,
               float alpha, float beta) {
  // int grid_size = m * n / kThreadPerBlock + 1;
  // hand_gemm_kernel<<<grid_size, kThreadPerBlock, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  // int BLOCK_M = 8;
  // int BLOCK_N = 128;
  // dim3 block(BLOCK_N, BLOCK_M);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M);

  // hand_gemm_kernel_blockmn<<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  // int BLOCK_M = 8;
  // int BLOCK_N = 128;
  // dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M);

  // hand_gemm_kernel_blockmn_4x4<<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  const int BLOCK_M = 64;
  const int BLOCK_N = 64;
  const int BLOCK_K = 32;
  dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  dim3 grid(n / BLOCK_N, m / BLOCK_M);

  hand_gemm_kernel_blockmn_4x4_shared<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  return cudaGetLastError();
}
