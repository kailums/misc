import torch

import triton
import triton.language as tl

import random


@triton.jit
def grouped_gemm_kernel(block_aligned_array, num_of_M,
        A_ptrs, B, C_ptrs,
        M_array, N, K,
        stride_am_array, stride_ak_array,
        stride_bk, stride_bn,
        stride_cm_array, stride_cn_array,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    ):
    pid = tl.program_id(0)
    # search for A pointer, M, C pointer of current pid
    new_pid = pid
    A = tl.load(A_ptrs).to(tl.pointer_type(B.dtype.element_ty))
    M = tl.load(M_array)
    C = tl.load(C_ptrs).to(tl.pointer_type(B.dtype.element_ty))
    stride_am = tl.load(stride_am_array)
    stride_ak = tl.load(stride_ak_array)
    stride_cm = tl.load(stride_cm_array)
    stride_cn = tl.load(stride_cn_array)

    for i in range(0, num_of_M):
        b_size = tl.load(block_aligned_array + i)
        if pid >= 0 and pid < b_size:
            # found
            A = tl.load(A_ptrs + i).to(tl.pointer_type(B.dtype.element_ty))
            M = tl.load(M_array + i)
            C = tl.load(C_ptrs + i).to(tl.pointer_type(B.dtype.element_ty))
            stride_am = tl.load(stride_am_array + i)
            stride_ak = tl.load(stride_ak_array + i)
            stride_cm = tl.load(stride_cm_array + i)
            stride_cn = tl.load(stride_cn_array + i)
            new_pid = pid
        pid -= b_size

    pid = new_pid
    # matrix multiplication
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


def triton_grouped_gemm(list_a, b):
    device = b.device
    a = list_a[0]
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    K, N = b.shape
    # allocates output
    out_ptrs = []
    a_ptrs = []
    m_sizes = []
    am_strides = []
    ak_strides = []
    cm_strides = []
    cn_strides = []
    block_aligned = []
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    list_c = []

    for a in list_a:
        M = a.shape[0]
        c = torch.zeros((M, N), device=device, dtype=a.dtype)
        list_c.append(c)
        out_ptrs.append(c.data_ptr())
        a_ptrs.append(a.data_ptr())
        m_sizes.append(M)
        am_strides.append(a.stride(0))
        ak_strides.append(a.stride(1))
        cm_strides.append(c.stride(0))
        cn_strides.append(c.stride(1))
        block_aligned.append(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

    # convert list into cuda tensors
    out_ptrs_tensor = torch.tensor(tuple(out_ptrs), dtype=torch.int64, device=device)
    a_ptrs_tensor = torch.tensor(tuple(a_ptrs), dtype=torch.int64, device=device)
    m_sizes_tensor = torch.tensor(tuple(m_sizes), dtype=torch.int32, device=device)
    am_strides_tensor = torch.tensor(tuple(am_strides), dtype=torch.int64, device=device)
    ak_strides_tensor = torch.tensor(tuple(ak_strides), dtype=torch.int64, device=device)
    cm_strides_tensor = torch.tensor(tuple(cm_strides), dtype=torch.int64, device=device)
    cn_strides_tensor = torch.tensor(tuple(cn_strides), dtype=torch.int64, device=device)
    block_aligned_tensor = torch.tensor(tuple(block_aligned), dtype=torch.int32, device=device)

    # launch kernel
    #grid = lambda META: (triton.cdiv(m_sizes[0], META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    grid = (sum(block_aligned), )
    grouped_gemm_kernel[grid](block_aligned_tensor, len(list_a),
                  a_ptrs_tensor, b, out_ptrs_tensor, 
                  m_sizes_tensor, N, K,
                  am_strides_tensor, ak_strides_tensor,
                  b.stride(0), b.stride(1),
                  cm_strides_tensor, cn_strides_tensor,
                  BLOCK_M=BLOCK_M,
                  BLOCK_N=BLOCK_N,
                  BLOCK_K=BLOCK_K,
                  GROUP_M=8,
                  EVEN_K=int(K % BLOCK_K == 0),
                  )
    return list_c


def torch_grouped_gemm(list_a, b):
    list_c = []
    for a in list_a:
        c = torch.matmul(a, b)
        list_c.append(c)
    return list_c


if __name__ == '__main__':
    batch=16
    M_start = 128
    M_end = 4096
    M = []
    for i in range(batch):
        m = random.randint(M_start, M_end)
        M.append(m)

    K = 1024
    N = 512

    device = torch.device(0)
    #dtype = torch.float16
    dtype = torch.float32
    B = torch.randn(K, N, device=device, dtype=dtype)

    A_list = []
    for m in M:
        a = torch.randn(m, K, device=device, dtype=dtype)
        A_list.append(a)

    # compare results
    triton_res = triton_grouped_gemm(A_list, B)
    torch_res = torch_grouped_gemm(A_list, B)
    for t1, t2 in zip(triton_res, torch_res):
        if torch.allclose(t1, t2):
            print('SAME')
        else:
            diff = abs(t1 - t2)
            print('max diff: ', diff.max(), ' rel-diff: ', (diff / t2).max())
            #print(t1)
            #print(t2)

    print('torch: ', triton.testing.do_bench(lambda: torch_grouped_gemm(A_list, B)))
    print('triton: ', triton.testing.do_bench(lambda: triton_grouped_gemm(A_list, B)))

