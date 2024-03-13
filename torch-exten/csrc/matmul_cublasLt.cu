#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CUBLAS_RETURN_IF_ERROR(expr) \
  do {                               \
    cublasStatus_t status = (expr);  \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      if (status == CUBLAS_STATUS_INVALID_VALUE) \
        std::cout << "call " << #expr << "failed, ret: invalid value" << std::endl; \
      else \
        std::cout << "call " << #expr << "failed, ret: " << status << std::endl; \
      return; \
    } \
  } while (0)

namespace torch_ext {
template <typename T, cublasComputeType_t COMPUTE_TYPE, cudaDataType DATA_TYPE>
void cublasLtMatmulHelper(
    cublasLtHandle_t lthandle,
    bool transa,
    bool transb,
    int m,
    int n,
    int k,
    float alpha,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    float beta,
    T *C,
    int ldc,
    cudaStream_t stream) {
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
  // set the transforms for A and B
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operationDesc, COMPUTE_TYPE, CUDA_R_32F));
  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  // create matrix descriptors, we are good with the details here so no need to set any extra attributes
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, DATA_TYPE, transa ? k : m, transa ? m : k, lda));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, DATA_TYPE, transb ? n : k, transb ? k : n, ldb));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, DATA_TYPE, m, n, ldc));

  // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)

  // allocate workspace
  uint64_t workspaceSize = 1024*1024*4;
  void* workspace = nullptr;
  // uint32_t align = 1;
  cudaMallocAsync(&workspace, workspaceSize, stream);
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align, sizeof(align)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align, sizeof(align)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align, sizeof(align)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align, sizeof(align)));

  // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
  // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoGetHeuristic(lthandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) {
    TORCH_CHECK(false, "cublasLt get no results");
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(lthandle,
                                    operationDesc,
                                    &alpha,
                                    A,
                                    Adesc,
                                    B,
                                    Bdesc,
                                    &beta,
                                    C,
                                    Cdesc,
                                    C,
                                    Cdesc,
                                    &heuristicResult.algo,
                                    workspace,
                                    workspaceSize,
                                    stream));
  cudaFreeAsync(workspace, stream);

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  return;
}

void cublasLtMatmul(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
  assert (A.dim() == 2 && B.dim() == 2 && C.dim() == 2);
  int M = A.size(0);
  int N = B.size(1);
  int K = A.size(1);
  assert (K == B.size(0) && M == C.size(0) && N == C.size(1));
  cublasLtHandle_t lthandle;
  CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&lthandle));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (A.dtype() == at::ScalarType::Float) {
    return cublasLtMatmulHelper<float, CUBLAS_COMPUTE_32F, CUDA_R_32F>(
        lthandle, false, false, N, M, K, 1.0f, B.data_ptr<float>(), N, A.data_ptr<float>(), K, 0.0f, C.data_ptr<float>(), N, stream);
  } else if (A.dtype() == at::ScalarType::Half) {
    return cublasLtMatmulHelper<at::Half, CUBLAS_COMPUTE_32F, CUDA_R_16F>(
        lthandle, false, false, N, M, K, 1.0f, B.data_ptr<at::Half>(), N, A.data_ptr<at::Half>(), K, 0.0f, C.data_ptr<at::Half>(), N, stream);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", A.dtype());
  }
  CUBLAS_RETURN_IF_ERROR(cublasLtDestroy(lthandle));
}

}  // namespace torch_ext
