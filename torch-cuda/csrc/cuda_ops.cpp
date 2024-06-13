#include <torch/extension.h>
#include <c10/util/Optional.h>

void fused_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key,
  torch::Tensor& value,
  float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "fused_attention",
    &fused_attention,
    "fused attention.");
}
