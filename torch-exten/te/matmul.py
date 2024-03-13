import torch
import te._C as _C

def matmul_cublasLt(A, B):
    output_shape = [A.shape[0], B.shape[1]]
    output = torch.zeros(output_shape, dtype=A.dtype, device=A.device)
    _C.matmul_cublasLt(A, B, output)
    return output
