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


// Helper function to print a matrix
template<typename T>
void print_matrix(std::string name, std::vector<T> &M, int rows, int cols) {
    std::cout << name << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << M[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
__global__ void _Split3InnerKernel(const int64_t size0_in_byte,
                                   const int64_t size1_in_byte,
                                   const int64_t size2_in_byte,
                                   const void* input_data,
                                   void* output_data0,
                                   void* output_data1,
                                   void* output_data2,
                                   const int64_t inner_size_in_byte) {
  // each block copy one row of input data
  auto size0 = size0_in_byte / sizeof(T);
  auto size1 = size1_in_byte / sizeof(T);
  auto size2 = size2_in_byte / sizeof(T);
  auto inner_size = inner_size_in_byte / sizeof(T);
  auto output0_vec = reinterpret_cast<T*>(output_data0) + blockIdx.x * size0;
  auto output1_vec = reinterpret_cast<T*>(output_data1) + blockIdx.x * size1;
  auto output2_vec = reinterpret_cast<T*>(output_data2) + blockIdx.x * size2;
  auto input_vec = reinterpret_cast<const T*>(input_data) + blockIdx.x * inner_size;
  // all size and pointer are aligned to sizeof(T)
  // so here use all threads in the block to do vectorized copy

  for (auto tid = threadIdx.x; tid < inner_size; tid += blockDim.x) {
    auto data = input_vec[tid];
    if (tid < size0) {
      output0_vec[tid] = data;
    } else if (tid < (size0 + size1)) {
      output1_vec[tid - size0] = data;
    } else {
      output2_vec[tid - size0 - size1] = data;
    }
  }
}

#define CUDA_LONG int32_t
constexpr int kNumThreadsPerBlock = 512;

void Split3Inner(cudaStream_t stream, const size_t element_size, const int64_t size0, const int64_t size1,
                   const int64_t size2, const void* input_data, void* output_data0, void* output_data1,
                   void* output_data2, const std::vector<int64_t>& input_shape) {
  CUDA_LONG outer_size = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      outer_size *= static_cast<CUDA_LONG>(input_shape[i]);
  }
  CUDA_LONG inner_size_in_byte = static_cast<CUDA_LONG>(input_shape[input_shape.size() - 1] * element_size);

  auto select = [](size_t value) {
    if (value % 16 == 0) {
      return 16;
    } else if (value % 8 == 0) {
      return 8;
    } else if (value % 4 == 0) {
      return 4;
    } else if (value % 2 == 0) {
      return 2;
    } else {
      return 1;
    }
  };

  auto input_v = reinterpret_cast<size_t>(input_data);
  auto output_v0 = reinterpret_cast<size_t>(output_data0);
  auto output_v1 = reinterpret_cast<size_t>(output_data1);
  auto output_v2 = reinterpret_cast<size_t>(output_data2);
  auto size0_in_byte = size0 * element_size;
  auto size1_in_byte = size1 * element_size;
  auto size2_in_byte = size2 * element_size;

  auto VEC_SIZE = std::min(select(size0_in_byte), std::min(select(size1_in_byte), select(size2_in_byte)));
  auto min_output_vec_size = std::min(select(output_v0), std::min(select(output_v1), select(output_v2)));
  VEC_SIZE = std::min(VEC_SIZE, std::min(select(input_v), min_output_vec_size));

  // determine threads based on the size of the output
  auto threadsPerBlock = kNumThreadsPerBlock;
  if ((inner_size_in_byte / VEC_SIZE) <= 128) {
    // use less threads when the size is small
    threadsPerBlock = 128;
  }

  switch (VEC_SIZE) {
#define CASE_ELEMENT_TYPE(type)                                                                       \
    std::cout << "select VEC_SIZE: " << VEC_SIZE << " grid: " << outer_size << " block: " << threadsPerBlock << std::endl; \
    _Split3InnerKernel<type><<<outer_size, threadsPerBlock, 0, stream>>>(                             \
                                                            size0_in_byte,                            \
                                                            size1_in_byte,                            \
                                                            size2_in_byte,                             \
                                                            input_data,        \
                                                            output_data0,      \
                                                            output_data1,      \
                                                            output_data2,      \
                                                            inner_size_in_byte)
    case 16:
      CASE_ELEMENT_TYPE(int4);
      break;
    case 8:
      CASE_ELEMENT_TYPE(int64_t);
      break;
    case 4:
      CASE_ELEMENT_TYPE(int32_t);
      break;
    case 2:
      CASE_ELEMENT_TYPE(int16_t);
      break;
    default:
      CASE_ELEMENT_TYPE(int8_t);
      break;
#undef CASE_ELEMENT_TYPE
  }

}

int main() {
    // Matrix dimensions
    int B = 4;
    int N = 93;

    using DTYPE = int8_t;

    // Host matrices
    std::vector<DTYPE> h_A(B*N);
    std::vector<int64_t> input_shape = {B, N};

    // Device matrices
    DTYPE *d_A;
    DTYPE *d_split1;
    DTYPE *d_split2;
    DTYPE *d_split3;

    size_t size_A = h_A.size() * sizeof(DTYPE);
    size_t size_split1 = N / 3;
    size_t size_split2 = N / 3;
    size_t size_split3 = N / 3;

    // Allocate memory for device matrices
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_split1, size_split1 * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&d_split2, size_split2 * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&d_split3, size_split3 * sizeof(DTYPE)));

    // Initialize host matrices with some values
    for (int i = 0; i < h_A.size(); i++) {
        h_A[i] = i + 1;
    }

    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));

    std::cout << "start running kernel" << std::endl;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Split3Inner(stream, sizeof(DTYPE), size_split1, size_split2, size_split3, d_A, d_split1, d_split2, d_split3, input_shape);
    cudaStreamSynchronize(stream);

    std::vector<DTYPE> h_hand_C(size_split1);
    CUDA_CHECK(cudaMemcpy(h_hand_C.data(), d_split1, size_split1 * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    print_matrix("C: ", h_hand_C, B, size_split1);

   // Free device memory
   CUDA_CHECK(cudaFree(d_A));
   CUDA_CHECK(cudaFree(d_split1));
   CUDA_CHECK(cudaFree(d_split2));
   CUDA_CHECK(cudaFree(d_split3));


   std::cout << "run done." << std::endl;

   return 0;
}
