#include <cuda_runtime.h>
#include <stdio.h>

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
__device__ inline void compute_float4_reg(float4 A, float4 B, float4* C) {
  C[0].x = fma(A.x, B.x, C[0].x);
  C[0].y = fma(A.x, B.y, C[0].y);
  C[0].z = fma(A.x, B.z, C[0].z);
  C[0].w = fma(A.x, B.w, C[0].w);

  C[1].x = fma(A.y, B.x, C[1].x);
  C[1].y = fma(A.y, B.y, C[1].y);
  C[1].z = fma(A.y, B.z, C[1].z);
  C[1].w = fma(A.y, B.w, C[1].w);

  C[2].x = fma(A.z, B.x, C[2].x);
  C[2].y = fma(A.z, B.y, C[2].y);
  C[2].z = fma(A.z, B.z, C[2].z);
  C[2].w = fma(A.z, B.w, C[2].w);

  C[3].x = fma(A.w, B.x, C[3].x);
  C[3].y = fma(A.w, B.y, C[3].y);
  C[3].z = fma(A.w, B.z, C[3].z);
  C[3].w = fma(A.w, B.w, C[3].w);
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

__device__ void atomic_add_float4(float* C, int idx, int n, int idy, float4* res, float alpha, float beta) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    atomicAdd(C+(idx + i)*n + idy + 0, res[i].x * alpha);
    atomicAdd(C+(idx + i)*n + idy + 1, res[i].y * alpha);
    atomicAdd(C+(idx + i)*n + idy + 2, res[i].z * alpha);
    atomicAdd(C+(idx + i)*n + idy + 3, res[i].w * alpha);
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
__device__ inline void load_gmem_to_smem_A(const float* ptr, int step, float* dst) {
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X / 4; j += blockDim.x) {
      reinterpret_cast<float4*>(dst + i * BLOCK_X)[j] = reinterpret_cast<const float4*>(ptr + i *step)[j];
      //dst[i*BLOCK_X + j] = ptr[i * step + j];
    }
  }
}

template<int BLOCK_Y, int BLOCK_X>
__device__ inline void load_gmem_to_smem_A_trans(const float* ptr, int step, float* dst) {
  // ptr is YxX
  // dst should be XxY
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X / 4; j += blockDim.x) {
      //reinterpret_cast<float4*>(dst + j * BLOCK_Y)[i] = reinterpret_cast<const float4*>(ptr + i *step)[j];
      float4 tmp = reinterpret_cast<const float4*>(ptr + i *step)[j];
      (dst + j * 4 * BLOCK_Y)[i] = tmp.x;
      (dst + (j * 4 +1) * BLOCK_Y)[i] = tmp.y;
      (dst + (j * 4 +2) * BLOCK_Y)[i] = tmp.z;
      (dst + (j * 4 +3) * BLOCK_Y)[i] = tmp.w;
    }
  }
}
template<int BLOCK_Y, int BLOCK_X>
__device__ inline void load_gmem_to_smem_B(const float* ptr, int step, float* dst) {
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
    load_gmem_to_smem_A<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + kk, lda, sm_A);
    load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + kk * ldb + block_x_offset, ldb, sm_B);
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

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M>
__global__ void hand_gemm_kernel_blockmn_4x4_shared_l2(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int group_size = gridDim.x * GROUP_M;
  int group_id = pid / group_size;
  int first_id = group_id * GROUP_M;
  int group_m_size = min(gridDim.y - first_id, GROUP_M);  // used for m is not dividable by GROUP_M
  int grid_m = first_id + pid % group_m_size;
  int grid_n = (pid % group_size) / group_m_size;

  int j = grid_n * BLOCK_N + threadIdx.x * 4;
  int i = grid_m * BLOCK_M + threadIdx.y * 4;
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  //   printf("x: %d, y: %d, grid_n: %d, grid_m: %d\n", blockIdx.x, blockIdx.y, grid_n, grid_m);
  // }
  if (i >= m || j >= n) return;

  __shared__ float sm_A[BLOCK_M * BLOCK_K];
  __shared__ float sm_B[BLOCK_K * BLOCK_N];

  float4 res[4] = {0, 0, 0, 0};
  float a_reg[4];
  float4 b_reg;

  int block_y_offset = grid_m * BLOCK_M;
  int block_x_offset = grid_n * BLOCK_N;

  for (int kk = 0; kk < k; kk += BLOCK_K) {
    // load memory from global to shared
    load_gmem_to_smem_A<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + kk, lda, sm_A);
    load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + kk * ldb + block_x_offset, ldb, sm_B);
    __syncthreads();
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      a_reg[0] = sm_A[threadIdx.y * 4 * BLOCK_K + kkk];
      a_reg[1] = sm_A[(threadIdx.y * 4 +1)* BLOCK_K + kkk];
      a_reg[2] = sm_A[(threadIdx.y * 4 +2)* BLOCK_K + kkk];
      a_reg[3] = sm_A[(threadIdx.y * 4 +3)* BLOCK_K + kkk];

      b_reg = reinterpret_cast<float4*>(sm_B + (BLOCK_N * kkk))[threadIdx.x];
      
      compute_float4(a_reg, b_reg, res);
    }
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }
  store_float4(C, i, ldc, j, res, alpha, beta);
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M>
__global__ void hand_gemm_kernel_blockmn_4x4_shared_l2_prefetch(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int group_size = gridDim.x * GROUP_M;
  int group_id = pid / group_size;
  int first_id = group_id * GROUP_M;
  int group_m_size = min(gridDim.y - first_id, GROUP_M);  // used for m is not dividable by GROUP_M
  int grid_m = first_id + pid % group_m_size;
  int grid_n = (pid % group_size) / group_m_size;

  int j = grid_n * BLOCK_N + threadIdx.x * 4;
  int i = grid_m * BLOCK_M + threadIdx.y * 4;

  if (i >= m || j >= n) return;

  __shared__ float sm_A[2][BLOCK_M * BLOCK_K];
  __shared__ float sm_B[2][BLOCK_K * BLOCK_N];

  float4 res[4] = {0, 0, 0, 0};
  float a_reg[4];
  float4 b_reg;

  int block_y_offset = grid_m * BLOCK_M;
  int block_x_offset = grid_n * BLOCK_N;

  int sm_id = 0;

  // prefetch gmem into smem for kk = 0
  load_gmem_to_smem_A<BLOCK_M, BLOCK_K>(A + block_y_offset*lda, lda, sm_A[sm_id]);
  load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + block_x_offset, ldb, sm_B[sm_id]);
  __syncthreads();

  for (int kk = 0; kk < k; kk += BLOCK_K) {
    // load memory from global to shared
    int next_sm_id = (sm_id + 1) % 2;
    if ((kk+BLOCK_K) < k) {
      load_gmem_to_smem_A<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + (kk + BLOCK_K), lda, sm_A[next_sm_id]);
      load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + (kk + BLOCK_K) * ldb + block_x_offset, ldb, sm_B[next_sm_id]);
      __syncthreads();
    }
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      a_reg[0] = sm_A[sm_id][threadIdx.y * 4 * BLOCK_K + kkk];
      a_reg[1] = sm_A[sm_id][(threadIdx.y * 4 +1)* BLOCK_K + kkk];
      a_reg[2] = sm_A[sm_id][(threadIdx.y * 4 +2)* BLOCK_K + kkk];
      a_reg[3] = sm_A[sm_id][(threadIdx.y * 4 +3)* BLOCK_K + kkk];

      b_reg = reinterpret_cast<float4*>(sm_B[sm_id] + (BLOCK_N * kkk))[threadIdx.x];
      
      compute_float4(a_reg, b_reg, res);
    }
    sm_id = next_sm_id;
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }
  store_float4(C, i, ldc, j, res, alpha, beta);
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M, int SPLIT_K>
__global__ void hand_gemm_kernel_blockmn_4x4_shared_l2_prefetch_splitK(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int group_size = gridDim.x * GROUP_M;
  int group_id = pid / group_size;
  int first_id = group_id * GROUP_M;
  int group_m_size = min(gridDim.y - first_id, GROUP_M);  // used for m is not dividable by GROUP_M
  int grid_m = first_id + pid % group_m_size;
  int grid_n = (pid % group_size) / group_m_size;

  int j = grid_n * BLOCK_N + threadIdx.x * 4;
  int i = grid_m * BLOCK_M + threadIdx.y * 4;

  if (i >= m || j >= n) return;

  __shared__ float sm_A[2][BLOCK_M * BLOCK_K];
  __shared__ float sm_B[2][BLOCK_K * BLOCK_N];

  float4 res[4] = {0, 0, 0, 0};
  float a_reg[4];
  float4 b_reg;

  int block_y_offset = grid_m * BLOCK_M;
  int block_x_offset = grid_n * BLOCK_N;

  int sm_id = 0;

  // prefetch gmem
  int k_step = k / SPLIT_K;
  int kk = blockIdx.z * k_step;
  int K_end = kk + k_step;

  load_gmem_to_smem_A<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + kk, lda, sm_A[sm_id]);
  load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + kk*ldb + block_x_offset, ldb, sm_B[sm_id]);
  __syncthreads();

  for (; kk < K_end; kk += BLOCK_K) {
    // load memory from global to shared
    int next_sm_id = (sm_id + 1) % 2;
    if ((kk+BLOCK_K) < K_end) {
      load_gmem_to_smem_A<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + (kk + BLOCK_K), lda, sm_A[next_sm_id]);
      load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + (kk + BLOCK_K) * ldb + block_x_offset, ldb, sm_B[next_sm_id]);
      __syncthreads();
    }
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      a_reg[0] = sm_A[sm_id][threadIdx.y * 4 * BLOCK_K + kkk];
      a_reg[1] = sm_A[sm_id][(threadIdx.y * 4 +1)* BLOCK_K + kkk];
      a_reg[2] = sm_A[sm_id][(threadIdx.y * 4 +2)* BLOCK_K + kkk];
      a_reg[3] = sm_A[sm_id][(threadIdx.y * 4 +3)* BLOCK_K + kkk];

      b_reg = reinterpret_cast<float4*>(sm_B[sm_id] + (BLOCK_N * kkk))[threadIdx.x];
      
      compute_float4(a_reg, b_reg, res);
    }
    sm_id = next_sm_id;
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }

  atomic_add_float4(C, i, ldc, j, res, alpha, beta);
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M, int SPLIT_K>
__global__ void hand_gemm_kernel_blockmn_4x4_shared_l2_prefetch_splitK_transA(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int group_size = gridDim.x * GROUP_M;
  int group_id = pid / group_size;
  int first_id = group_id * GROUP_M;
  int group_m_size = min(gridDim.y - first_id, GROUP_M);  // used for m is not dividable by GROUP_M
  int grid_m = first_id + pid % group_m_size;
  int grid_n = (pid % group_size) / group_m_size;

  int j = grid_n * BLOCK_N + threadIdx.x * 4;
  int i = grid_m * BLOCK_M + threadIdx.y * 4;

  if (i >= m || j >= n) return;

  __shared__ float sm_A[2][BLOCK_M * BLOCK_K];
  __shared__ float sm_B[2][BLOCK_K * BLOCK_N];

  float4 res[4] = {0, 0, 0, 0};
  float4 a_reg;
  float4 b_reg;

  int block_y_offset = grid_m * BLOCK_M;
  int block_x_offset = grid_n * BLOCK_N;

  int sm_id = 0;

  // prefetch gmem
  int k_step = k / SPLIT_K;
  int kk = blockIdx.z * k_step;
  int K_end = kk + k_step;

  load_gmem_to_smem_A_trans<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + kk, lda, sm_A[sm_id]);
  load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + kk*ldb + block_x_offset, ldb, sm_B[sm_id]);
  __syncthreads();

  for (; kk < K_end; kk += BLOCK_K) {
    // load memory from global to shared
    int next_sm_id = (sm_id + 1) % 2;
    if ((kk+BLOCK_K) < K_end) {
      load_gmem_to_smem_A_trans<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + (kk + BLOCK_K), lda, sm_A[next_sm_id]);
      load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + (kk + BLOCK_K) * ldb + block_x_offset, ldb, sm_B[next_sm_id]);
      __syncthreads();
    }
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      a_reg = reinterpret_cast<float4*>(sm_A[sm_id] + (BLOCK_M * kkk))[threadIdx.y];
      b_reg = reinterpret_cast<float4*>(sm_B[sm_id] + (BLOCK_N * kkk))[threadIdx.x];
      
      compute_float4_reg(a_reg, b_reg, res);
    }
    sm_id = next_sm_id;
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }

  atomic_add_float4(C, i, ldc, j, res, alpha, beta);
}

__device__ inline void compute_float4_reg_8x8(float4* A, float4* B, float4* C) {
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      // TODO: here the res is not correct, should set stride for computing sub-matrix of C
      compute_float4_reg(A[i], B[j], &C[i* 8 + j * 4]);
    }
  }
}

__device__ void atomic_add_float4_8x8(float* C, int idy, int step, int idx, float4* res, float alpha, float beta) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 0, res[i* 8 + j*4 + 0].x * alpha);
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 1, res[i* 8 + j*4 + 0].y * alpha);
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 2, res[i* 8 + j*4 + 0].z * alpha);
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 3, res[i* 8 + j*4 + 0].w * alpha);

      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 4, res[i* 8 + j*4 + 1].x * alpha);
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 5, res[i* 8 + j*4 + 1].y * alpha);
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 6, res[i* 8 + j*4 + 1].z * alpha);
      atomicAdd(C+(idy + i*4 + j*2)*step + idx + 7, res[i* 8 + j*4 + 1].w * alpha);

      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 0, res[i* 8 + j*4 + 2].x * alpha);
      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 1, res[i* 8 + j*4 + 2].y * alpha);
      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 2, res[i* 8 + j*4 + 2].z * alpha);
      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 3, res[i* 8 + j*4 + 2].w * alpha);

      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 4, res[i* 8 + j*4 + 3].x * alpha);
      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 5, res[i* 8 + j*4 + 3].y * alpha);
      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 6, res[i* 8 + j*4 + 3].z * alpha);
      atomicAdd(C+(idy + i*4 + j*2 + 1)*step + idx + 7, res[i* 8 + j*4 + 3].w * alpha);
    }
  }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M, int SPLIT_K>
__global__ void hand_gemm_kernel_blockmn_shared_l2_prefetch_splitK_transA_8x8(int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  // each thread compute 4x4 result
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int group_size = gridDim.x * GROUP_M;
  int group_id = pid / group_size;
  int first_id = group_id * GROUP_M;
  int group_m_size = min(gridDim.y - first_id, GROUP_M);  // used for m is not dividable by GROUP_M
  int grid_m = first_id + pid % group_m_size;
  int grid_n = (pid % group_size) / group_m_size;

  int j = grid_n * BLOCK_N + threadIdx.x * 8;
  int i = grid_m * BLOCK_M + threadIdx.y * 8;

  if (i >= m || j >= n) return;

  __shared__ float sm_A[2][BLOCK_M * BLOCK_K];
  __shared__ float sm_B[2][BLOCK_K * BLOCK_N];

  float4 res[16] = {0};
  float4 a_reg[2];
  float4 b_reg[2];

  int block_y_offset = grid_m * BLOCK_M;
  int block_x_offset = grid_n * BLOCK_N;

  int sm_id = 0;

  // prefetch gmem
  int k_step = k / SPLIT_K;
  int kk = blockIdx.z * k_step;
  int K_end = kk + k_step;

  load_gmem_to_smem_A_trans<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + kk, lda, sm_A[sm_id]);
  load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + kk*ldb + block_x_offset, ldb, sm_B[sm_id]);
  __syncthreads();

  for (; kk < K_end; kk += BLOCK_K) {
    // load memory from global to shared
    int next_sm_id = (sm_id + 1) % 2;
    if ((kk+BLOCK_K) < K_end) {
      load_gmem_to_smem_A_trans<BLOCK_M, BLOCK_K>(A + block_y_offset*lda + (kk + BLOCK_K), lda, sm_A[next_sm_id]);
      load_gmem_to_smem_B<BLOCK_K, BLOCK_N>(B + (kk + BLOCK_K) * ldb + block_x_offset, ldb, sm_B[next_sm_id]);
      __syncthreads();
    }
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      a_reg[0] = reinterpret_cast<float4*>(sm_A[sm_id] + (BLOCK_M * kkk))[threadIdx.y * 2];
      a_reg[1] = reinterpret_cast<float4*>(sm_A[sm_id] + (BLOCK_M * kkk))[threadIdx.y * 2 + 1];
      b_reg[0] = reinterpret_cast<float4*>(sm_B[sm_id] + (BLOCK_N * kkk))[threadIdx.x * 2];
      b_reg[1] = reinterpret_cast<float4*>(sm_B[sm_id] + (BLOCK_N * kkk))[threadIdx.x * 2 + 1];
      
      compute_float4_reg_8x8(a_reg, b_reg, res);
    }
    sm_id = next_sm_id;
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }

  atomic_add_float4_8x8(C, i, ldc, j, res, alpha, beta);
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

  // const int BLOCK_M = 64;
  // const int BLOCK_N = 64;
  // const int BLOCK_K = 64;
  // dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M);

  // //const int kConfigSharedMem =65535;
  // //cudaFuncSetAttribute(hand_gemm_kernel_blockmn_4x4_shared<BLOCK_M, BLOCK_N, BLOCK_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, kConfigSharedMem);
  // //cudaFuncSetAttribute(hand_gemm_kernel_blockmn_4x4_shared<BLOCK_M, BLOCK_N, BLOCK_K>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  // hand_gemm_kernel_blockmn_4x4_shared<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  // const int BLOCK_M = 64;
  // const int BLOCK_N = 32;
  // const int BLOCK_K = 128;
  // const int GROUP_M=4;
  // dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M);

  // hand_gemm_kernel_blockmn_4x4_shared_l2<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  // const int BLOCK_M = 64;
  // const int BLOCK_N = 64;
  // const int BLOCK_K = 32;
  // const int GROUP_M = 4;
  // dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M);

  // hand_gemm_kernel_blockmn_4x4_shared_l2_prefetch<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  // const int BLOCK_M = 64;
  // const int BLOCK_N = 64;
  // const int BLOCK_K = 32;
  // const int GROUP_M = 4;
  // const int SPLIT_K = 32;
  // dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M, SPLIT_K);

  // hand_gemm_kernel_blockmn_4x4_shared_l2_prefetch_splitK<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, SPLIT_K><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  // const int BLOCK_M = 64;
  // const int BLOCK_N = 64;
  // const int BLOCK_K = 32;
  // const int GROUP_M = 4;
  // const int SPLIT_K = 32;
  // dim3 block(BLOCK_N / 4, BLOCK_M / 4);
  // dim3 grid(n / BLOCK_N, m / BLOCK_M, SPLIT_K);

  // hand_gemm_kernel_blockmn_4x4_shared_l2_prefetch_splitK_transA<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, SPLIT_K><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  const int BLOCK_M = 64;
  const int BLOCK_N = 128;
  const int BLOCK_K = 16;
  const int GROUP_M = 4;
  const int SPLIT_K = 8;
  dim3 block(BLOCK_N / 8, BLOCK_M / 8);
  dim3 grid(n / BLOCK_N, m / BLOCK_M, SPLIT_K);

  hand_gemm_kernel_blockmn_shared_l2_prefetch_splitK_transA_8x8<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, SPLIT_K><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  return cudaGetLastError();
}
