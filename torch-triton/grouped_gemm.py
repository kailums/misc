import torch

import triton
import triton.language as tl

import random


@triton.jit
def grouped_gemm_kernel(block_aligned_array, num_of_matrix,
        M_array, N, K,
        array_alpha,
        A_ptrs, ldas,
        B_ptrs, ldb,
        array_beta,
        C_ptrs, ldcs,
        D_ptrs, ldds,
        DTYPE: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
        TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    ):
    pid = tl.program_id(0)
    # search for A pointer, M, C pointer of current pid
    new_pid = pid
    A = tl.load(A_ptrs).to(tl.pointer_type(DTYPE))
    lda = tl.load(ldas)
    B = tl.load(B_ptrs).to(tl.pointer_type(DTYPE))
    ldb = tl.load(ldbs)
    M = tl.load(M_array)
    C = tl.load(C_ptrs).to(tl.pointer_type(DTYPE))
    ldc = tl.load(ldcs)
    D = tl.load(D_ptrs).to(tl.pointer_type(DTYPE))
    ldd = tl.load(ldds)
    alpha = tl.load(array_alpha)
    beta = tl.load(array_beta)

    for i in range(0, num_of_matrix):
        b_size = tl.load(block_aligned_array + i)
        if pid >= 0 and pid < b_size:
            # found
            A = tl.load(A_ptrs + i).to(tl.pointer_type(DTYPE))
            lda = tl.load(ldas + i)
            B = tl.load(B_ptrs + i).to(tl.pointer_type(DTYPE))
            ldb = tl.load(ldbs + i)
            M = tl.load(M_array + i)
            C = tl.load(C_ptrs + i).to(tl.pointer_type(DTYPE))
            ldc = tl.load(ldcs + i)
            D = tl.load(D_ptrs + i).to(tl.pointer_type(DTYPE))
            ldd = tl.load(ldds + i)
            alpha = tl.load(array_alpha + i)
            beta = tl.load(array_beta + i)

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
    if TRANS_A == 1:
        A = A + (ram[:, None] * lda + rk[None, :])
    else:
        A = A + (ram[None, :] * lda + rk[:, None])

    if TRANS_B == 1:
        B = B + (rk[:, None] * ldb + rbn[None, :])
    else:
        B = B + (rk[None, :] * ldb + rbn[:, None])

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
    C = C + (rm[None, :] * ldc + rn[:, None])
    mask = (rm < M)[None, :] & (rn < N)[:, None]
    tl.store(C, acc, mask=mask)


def triton_grouped_gemm(array_trans_a, array_trans_b, array_alpha, array_a, array_b, array_beta, array_c):
    """
    grouped gemm's signature is (trans_a, trans_b
                             vector<> m, vector<> n, vector<> k
                             vector<> alpha,
                             vector<> A, vector<> lda, vector<> B, vector<> ldb,
                             vector<> beta,
                             vector<> C, vector<> ldc,
                             vector<> D, vector<> ldd,
                             int gemm_count,
                             )
    where D[i] = alpha * A[i]*B[i] + beta * C[i], i = [0, gemm_count).
    A,B,C,D are in column-major format, then lda, ldb is the number of elements in one line.

    Here we use a list of tensor: array_a, array_b, array_c for vector of A, B, C pointers, and use the tensor shape for m, n, k

    To simplify the problem, we do some assumptions:
        1. all n in vector<> n are same, k in vector<> k are same, only m is variant. 
        2. all tensor have same layout, then only need 1 trans_a and 1 trans_b
        3. C has already been added into D with shape (m,n), so we compute C[i] = alpha * A[i]B[i] + beta *C[i]

    """
    a = array_a[0]
    device = a.device

    # allocates output
    a_ptrs = []
    m_sizes = []
    ldas = []
    b_ptrs = []
    c_ptrs = []
    ldcs = []  # same as ldas

    trans_a = array_trans_a[0]
    trans_b = array_trans_b[0]
    K = array_b[0].shape[0] if trans_b == 0 else array_b[0].shape[1]
    N = array_b[0].shape[1] if trans_b == 0 else array_b[0].shape[0]
    ldb = N if trans_b == 1 else K

    block_aligned = []
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    array_d = []

    for a,b,c in zip(array_a, array_b, array_c):
        M = a.shape[0] if trans_a == 0 else a.shape[1]
        lda = a.shape[1] if trans_a == 0 else a.shape[0]
        a_ptrs.append(a.data_ptr())
        m_sizes.append(M)
        ldas.append(lda)
        b_ptrs.append(b.data_ptr())
        c_ptrs.append(c.data_ptr())

        block_aligned.append(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

    # convert list into cuda tensors
    a_ptrs_tensor = torch.tensor(tuple(a_ptrs), dtype=torch.int64, device=device)
    m_sizes_tensor = torch.tensor(tuple(m_sizes), dtype=torch.int32, device=device)
    alpha_tensor = torch.tensor(tuple(array_alpha), dtype=torch.float32, device=device)
    beta_tensor = torch.tensor(tuple(array_beta), dtype=torch.float32, device=device)
    ldas_tensor = torch.tensor(tuple(ldas), dtype=torch.int32, device=device)
    b_ptrs_tensor = torch.tensor(tuple(b_ptrs), dtype=torch.int64, device=device)
    c_ptrs_tensor = torch.tensor(tuple(c_ptrs), dtype=torch.int64, device=device)

    block_aligned_tensor = torch.tensor(tuple(block_aligned), dtype=torch.int32, device=device)

    # launch kernel
    #grid = lambda META: (triton.cdiv(m_sizes[0], META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    grid = (sum(block_aligned), )
    grouped_gemm_kernel[grid](block_aligned_tensor, len(array_a),
                  m_sizes_tensor, N, K,
                  alpha_tensor,
                  a_ptrs_tensor, ldas_tensor,
                  b_ptrs_tensor, ldb,
                  beta_tensor,
                  c_ptrs_tensor, ldas_tensor,  # ldc is same as lda, use ldas for ldcs
                  c_ptrs_tensor, ldas_tensor,  # re-use c_ptr as output d ptr.
                  DTYPE=tl.float32 if array_a[0].dtype == torch.float32 else tl.float16,
                  ACC_DTYPE=tl.float32,
                  TRANS_A=trans_a, TRANS_B=trans_b,  # 0 for N, 1 for T
                  BLOCK_M=BLOCK_M,
                  BLOCK_N=BLOCK_N,
                  BLOCK_K=BLOCK_K,
                  GROUP_M=8,
                  EVEN_K=int(K % BLOCK_K == 0),
                  )
    return array_c


def torch_grouped_gemm(list_a, b):
    list_c = []
    for a in list_a:
        c = torch.matmul(a, b)
        list_c.append(c)
    return list_c

def test_speed(M_list, K, N, device, dtype):
    batch = len(M_list)
    max_m = max(M_list)
    aligned = 16
    M = (max_m // aligned + 1) * aligned

    # generate input data for triton
    B = torch.randn(K, N, device=device, dtype=dtype)
    A_triton = []
    for m in M_list:
        A_triton.append(torch.randn(m, K, device=device, dtype=dtype))

    # generate aligned input data for torch
    A_torch = torch.randn(batch, M, K, device=device, dtype=dtype)

    print('torch: ', triton.testing.do_bench(lambda: torch.matmul(A_torch, B)))
    print('triton: ', triton.testing.do_bench(lambda: triton_grouped_gemm(A_triton, B)))


if __name__ == '__main__':
    batch=16
    M_start = 128
    M_end = 4096
    M = []
    for i in range(batch):
        m = random.randint(M_start, M_end)
        M.append(m)

    print('M: ', M)
    K = 1024
    N = 512

    device = torch.device(0)
    dtype = torch.float16
    #dtype = torch.float32
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
            print('shape: ', t1.shape, 'max diff: ', diff.max(), ' rel-diff: ', (diff / t2).max())

    test_speed(M, K, N, device, dtype)

