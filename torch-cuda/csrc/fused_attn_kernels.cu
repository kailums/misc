#include <torch/extension.h>
#include <c10/util/Optional.h>

__global__ void fused_attention_kernel(
    half *out,
    const half *query,
    const half *key,
    const half *value,
    const float scale,
    const int num_heads,
    const int query_stride,
    const int key_stride,
    const int value_stride,
    const int seq_len,
    const int context_len,
    const int head_dim) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const auto *query_ptr = query + batch * num_heads * query_stride + head * query_stride;
  const auto *key_ptr = key + batch * num_heads * key_stride + head * key_stride;
  const auto *value_ptr = value + batch * num_heads * value_stride + head * value_stride;
}

void fused_attention(
  torch::Tensor& out,
  torch::Tensor& query,  // shape is (batch, num_heads, seq_len, head_dim)
  torch::Tensor& key,    // shape is (batch, num_heads, context_len, head_dim)
  torch::Tensor& value,  // shape is (batch, num_heads, context_len, head_dim)
  float scale) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int batch_size = query.size(0);
  const int num_heads = query.size(1);
  const int seq_len = query.size(2);
  const int head_dim = query.size(3);

  const int context_len = key.size(2);
  constexpr int NUM_THREADS = 128;

  dim3 grid(batch_size, num_heads);
  dim3 block(NUM_THREADS);

  fused_attention_kernel<<<grid, block, 0, stream>>>(
      out.data_ptr<at::Half>(),
      query.data_ptr<at::Half>(),
      key.data_ptr<at::Half>(),
      value.data_ptr<at::Half>(),
      scale);
}
