#include <torch/extension.h>
namespace torch_ext {
void cublasLtMatmul(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C);
} // namespace torch_ext


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Attention ops
  m.def(
    "matmul_cublasLt",
    &torch_ext::cublasLtMatmul,
    "Compute matmul using cublasLt.");
  
}
