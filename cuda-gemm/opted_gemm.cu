#include <cuda_runtime.h>
#include <stdio.h>

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
__device__ inline void load_gmem_to_smem_B(const float* ptr, int step, float* dst) {
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X / 4; j += blockDim.x) {
      reinterpret_cast<float4*>(dst + i * BLOCK_X)[j] = reinterpret_cast<const float4*>(ptr + i *step)[j];
    }
  }
}

template<int BLOCK_Y, int BLOCK_X>
__device__ inline void load_gmem_to_smem(float* ptr, int step, float* dst) {
  int tid = threadIdx.y * blockDim.y + threadIdx.x;
  int x_stride = BLOCK_X / 4;
  int row_id = tid / x_stride;
  int col_id = tid % x_stride;
  int row_stride = blockDim.y * blockDim.x / x_stride;
  #pragma unroll
  for (int i = 0; i < BLOCK_Y; i += row_stride) {
    int row = i + row_id;
    reinterpret_cast<float4*>(dst + row * BLOCK_X)[col_id] = reinterpret_cast<float4*>(ptr + row * step)[col_id];
  }
}


template<int STEP>
__device__ inline void compute_reg_8x8(float* A, float* B, float (&C)[STEP][STEP]) {
  #pragma unroll
  for (int i = 0; i < STEP; i++) {
    #pragma unroll
    for (int j = 0; j < STEP; j++) {
      C[i][j] = fma(A[i], B[j], C[i][j]);
    }
  }
}

template<int STEP>
__device__ inline void store_reg_to_gmem(float* c, int ldc, float (&reg)[STEP][STEP], int m_offset, int n_offset, float alpha, float beta) {
  auto* c_ptr = c + m_offset * ldc + n_offset;
  #pragma unroll
  for (int i = 0; i < STEP; ++i) {
    #pragma unroll
    for (int j = 0; j < STEP; ++j) {
      int row = i * blockDim.y + threadIdx.y;
      int col = j * blockDim.x + threadIdx.x;
      c_ptr[row * ldc + col] = reg[i][j] * alpha + c_ptr[row *ldc + col] * beta;
    }
  }
}
template<int BLOCK_Y, int BLOCK_X>
__device__ inline void load_gmem_to_smem_A_trans(float* ptr, int step, float* dst, int dst_step) {
  // ptr is YxX
  // dst should be XxY
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X / 4; j += blockDim.x) {
      float4 tmp = reinterpret_cast<float4*>(ptr + i *step)[j];
      (dst + j * 4 * dst_step)[i] = tmp.x;
      (dst + (j * 4 +1) * dst_step)[i] = tmp.y;
      (dst + (j * 4 +2) * dst_step)[i] = tmp.z;
      (dst + (j * 4 +3) * dst_step)[i] = tmp.w;
    }
  }
}

template<int STEP>
__device__ inline void load_smem_to_reg8(float* sm_ptr, int idx, int step, float* reg) {
  #pragma unroll
  for (int i = 0; i < STEP; ++i) {
    reg[i] = sm_ptr[idx];
    idx += step;
  }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int GROUP_M, int STEP>
__global__ void hand_gemm_kernel_blockmn_shared_l2_prefetch_transA_8x8(int m, int n, int k, float* __restrict__ A, int lda, float* __restrict__ B, int ldb, float* __restrict__ C, int ldc, float alpha, float beta) {
  // every block calc a size of BLOCK_M * BLOCK_N
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int group_size = gridDim.x * GROUP_M;
  int group_id = pid / group_size;
  int first_id = group_id * GROUP_M;
  int group_m_size = min(gridDim.y - first_id, GROUP_M);  // used for m is not dividable by GROUP_M
  int grid_m = first_id + pid % group_m_size;
  int grid_n = (pid % group_size) / group_m_size;

  int n_offset = grid_n * BLOCK_N;
  int m_offset = grid_m * BLOCK_M;

  __shared__ float sm_A[BLOCK_K][BLOCK_M];
  __shared__ float sm_B[BLOCK_K][BLOCK_N];

  float res[STEP][STEP] = {0};
  float a_reg[STEP];
  float b_reg[STEP];

  // prefetch gmem
  auto* a_ptr = A + m_offset*lda;
  auto* b_ptr = B + n_offset;

  for (int kk = 0; kk < k; kk += BLOCK_K) {
    // load memory from global to shared
    // load_gmem_to_smem_A_trans<BLOCK_M, BLOCK_K>(a_ptr, lda, &sm_A[0][0], BLOCK_M);
    load_gmem_to_smem<BLOCK_M, BLOCK_K>(a_ptr, lda, &sm_A[0][0]);
    a_ptr += BLOCK_K;
    load_gmem_to_smem<BLOCK_K, BLOCK_N>(b_ptr, ldb, &sm_B[0][0]);
    b_ptr += BLOCK_K * ldb;
    __syncthreads();
    
    #pragma unroll
    for (int kkk = 0; kkk < BLOCK_K; ++kkk) {
      // load_smem_to_reg8<STEP>(sm_A[kkk], threadIdx.y, blockDim.y, a_reg);
      // load_smem_to_reg8<STEP>(sm_B[kkk], threadIdx.x, blockDim.x, b_reg);
      // 
      // compute_reg_8x8<STEP>(a_reg, b_reg, res);
      #pragma unroll
      for (int i = 0; i < STEP; i++) {
        #pragma unroll
        for (int j = 0; j < STEP; j++) {
          // res[i][j] = fma(sm_A[kkk][i * blockDim.y + threadIdx.y], sm_B[kkk][j * blockDim.x + threadIdx.x], res[i][j]);
          res[i][j] += sm_A[i * blockDim.y + threadIdx.y][kkk] * sm_B[kkk][j * blockDim.x + threadIdx.x];
        }
      }
    }
    // need to sync before next load gmem to smem in case of writing smem before compute has done.
    __syncthreads();
  }

  store_reg_to_gmem<STEP>(C, ldc, res, m_offset, n_offset, alpha, beta);
}

cudaError_t hand_gemm_opted(cudaStream_t stream,
               int m, int n, int k, 
               float* A, int lda,
               float* B, int ldb,
               float* C, int ldc,
               float alpha, float beta) {
  const int BLOCK_M = 64;
  const int BLOCK_N = 64;
  const int BLOCK_K = 64;
  const int GROUP_M = 4;
  const int STEP = 4;
  dim3 block(BLOCK_N / STEP, BLOCK_M / STEP);
  dim3 grid(n / BLOCK_N, m / BLOCK_M);

  hand_gemm_kernel_blockmn_shared_l2_prefetch_transA_8x8<BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, STEP><<<grid, block, 0, stream>>>(m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);

  return cudaGetLastError();
}
