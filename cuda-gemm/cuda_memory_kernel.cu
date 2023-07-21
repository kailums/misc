#include <cuda_runtime.h>
#include <stdio.h>


__global__ void global_memory_copy(float* dst, float* src, int size) {
  int block_size = blockDim.x * blockDim.y;
  int i = blockIdx.x * block_size + threadIdx.y * blockDim.x + threadIdx.x + 4;
  // DRAM throughput drop from 87.9% to 86.2% with offset is not dividable by 32
  if (i >= size) return;

  dst[i] = src[i];
}

template<int BLOCK_Y, int BLOCK_X>
__device__ void load_gmem_to_smem(const float* ptr, int step, float* dst) {
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X/4; j += blockDim.x) {
      reinterpret_cast<float4*>(dst + i * BLOCK_X)[j] = reinterpret_cast<const float4*>(ptr + i *step)[j];
    }
  }
}

template<int BLOCK_Y, int BLOCK_X>
__device__ void load_gmem_to_smem_trans(const float* ptr, int step, float* dst, int dst_step) {
  #pragma unroll
  for (int i = threadIdx.y; i < BLOCK_Y; i += blockDim.y) {
    #pragma unroll
    for (int j = threadIdx.x; j < BLOCK_X/4; j += blockDim.x) {
      float4 data = reinterpret_cast<const float4*>(ptr + i *step)[j];
      (dst + (j * 4) * dst_step)[i] = data.x;
      (dst + (j * 4 + 1) * dst_step)[i] = data.y;
      (dst + (j * 4 + 2) * dst_step)[i] = data.z;
      (dst + (j * 4 + 3) * dst_step)[i] = data.w;
    }
    // #pragma unroll
    // for (int j = threadIdx.x; j < BLOCK_X; j += blockDim.x) {
    //   (dst + j * BLOCK_Y)[i] = (ptr + i * step)[j];
    // }
    // #pragma unroll
    // for (int j = threadIdx.x; j < BLOCK_X/2; j += blockDim.x) {
    //   float2 data = reinterpret_cast<const float2*>(ptr + i *step)[j];
    //   (dst + (j * 2) * BLOCK_Y)[i] = data.x;
    //   (dst + (j * 2 + 1) * BLOCK_Y)[i] = data.y;
    // }
  }
}


template<int BLOCK_M, int BLOCK_K>
__global__ void gmem_to_smem(float* src, int m, int k, float* dst) {
  __shared__ float A[BLOCK_M][BLOCK_K];
  int offset = blockIdx.x * BLOCK_M * k;
  auto* src_ptr = src + offset;
  float4 res = {0};
  const int trans_step = BLOCK_M + 1;
  __shared__ float A_trans[BLOCK_K][trans_step];
  for (int kk = 0; kk < k; kk += BLOCK_K) {
    //load_gmem_to_smem<BLOCK_M, BLOCK_K>(src_ptr + kk, k, &(A[0][0]));
    load_gmem_to_smem_trans<BLOCK_M, BLOCK_K>(src_ptr + kk, k, &(A_trans[0][0]), trans_step);
    __syncthreads();

    int m_offset = threadIdx.y;
    for (int i = 0; i < BLOCK_K; ++i) {
      res.x += A[m_offset][i] * A_trans[i][m_offset];
      auto mo = m_offset + blockDim.y;
      res.y += A[mo][i] * A_trans[i][mo];
      mo += blockDim.y;
      res.z += A[mo][i] * A_trans[i][mo];
      mo += blockDim.y;
      res.w += A[mo][i] * A_trans[i][mo];
    }
  }
  *reinterpret_cast<float4*>(dst + offset + threadIdx.x * 4) = res;
}

void call_memory_kernel(float* dev_dst, float* dev_src, int size, cudaStream_t stream) {
  int m = 2048;
  int k = 1024;

  // int block_size = 256;
  // dim3 block(32, block_size / 32);
  // dim3 grid(size / block_size, 1);

  // global_memory_copy<<<grid, block, 0, stream>>>(dev_dst, dev_src, size);
  
  const int BLOCK_M = 64;
  const int BLOCK_K = 64;
  int block_size = 256;
  dim3 block(16, 16);
  dim3 grid(m / BLOCK_M, 1);

  gmem_to_smem<BLOCK_M, BLOCK_K><<<grid, block, 0, stream>>>(dev_src, m, k, dev_dst);

}
