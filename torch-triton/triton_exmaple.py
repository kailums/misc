import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)   # 1D grid size
    block_start = pid * BLOCK_SIZE

    # BLOCK_SIZE should be tl.constexpr for arange.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    y = a + b

    tl.store(out_ptr + offsets, y, mask=mask)


def add(a : torch.Tensor, b: torch.Tensor):
    n_elements = a.numel()
    out = torch.empty_like(a)

    # grid should be a tuple or Callable[meta] that returns a tuple
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=128)

    return out

@triton.jit
def softmax_kernel(out_ptr, in_ptr, out_stride, in_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    in_mask = col_offset < n_cols
    in_offset = pid * in_stride + col_offset
    in_data = tl.load(in_ptr + in_offset, mask=in_mask, other=-float('inf'))
    in_max = in_data - tl.max(in_data, axis=0)
    in_max = tl.exp(in_max)
    in_sum = tl.sum(in_max, axis=0)
    res = in_max / in_sum

    out_offset = pid * out_stride + col_offset
    out_mask = col_offset < n_cols
    tl.store(out_ptr + out_offset, res, mask=out_mask)

def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4

    if BLOCK_SIZE > 2048:
        num_warps = 8
    if BLOCK_SIZE > 4096:
        num_warps = 16

    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, y.stride(0), x.stride(0), n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # you can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    M, KA  = a.shape
    KB, N = b.shape
    assert KA == KB

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: ((triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), ))

    matmul_kernel[grid](
            a, b, c, M, N, KA,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=16,
            GROUP_SIZE_M=16
        )
    return c



def main(args=None):
    device = torch.device(0)
    torch.manual_seed(0)
    a_shape = (1024, 2048)
    b_shape = (2048, 512)
    a = torch.randn(a_shape).to(device).half()
    b = torch.randn(b_shape).to(device).half()

    #triton_out = add(a, b)
    #torch_out = a + b

    #triton_out = softmax(a)
    #torch_out = torch.softmax(a, axis=1)

    triton_out = matmul(a, b)
    torch_out = torch.matmul(a,b)

    if torch.allclose(triton_out, torch_out):
        print('result is SAME')
    else:
        diff = abs(torch_out - triton_out)
        print('not SAME, diff: ', diff.max())


if __name__ == '__main__':
    main()

