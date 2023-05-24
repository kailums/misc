import torch

import triton
import triton.language as tl

import random


@triton.jit
def grouped_gemm_kernel(block_aligned_array, num_of_matrix,
        m_array, n_array, K,
        array_alpha,
        a_ptrs, ldas,
        b_ptrs, ldbs,
        array_beta,
        c_ptrs, ldcs,
        d_ptrs, ldds,
        DTYPE: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
        TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    ):
    pid = tl.program_id(0)

    # reset DTYPE, because it is tl.constexpr, can't be used in tl.pointer_type
    if DTYPE == tl.constexpr(tl.float16):
        DTYPE = tl.float16
    else:
        DTYPE = tl.float32

    # search for A pointer, M, C pointer of current pid
    new_pid = pid
    A = tl.load(a_ptrs).to(tl.pointer_type(DTYPE))
    lda = tl.load(ldas)
    B = tl.load(b_ptrs).to(tl.pointer_type(DTYPE))
    ldb = tl.load(ldbs)
    M = tl.load(m_array)
    N = tl.load(n_array)
    C = tl.load(c_ptrs).to(tl.pointer_type(DTYPE))
    ldc = tl.load(ldcs)
    D = tl.load(d_ptrs).to(tl.pointer_type(DTYPE))
    ldd = tl.load(ldds)
    alpha = tl.load(array_alpha)
    beta = tl.load(array_beta)

    for i in range(0, num_of_matrix):
        b_size = tl.load(block_aligned_array + i)
        if pid >= 0 and pid < b_size:
            # found
            A = tl.load(a_ptrs + i).to(tl.pointer_type(DTYPE))
            lda = tl.load(ldas + i)
            B = tl.load(b_ptrs + i).to(tl.pointer_type(DTYPE))
            ldb = tl.load(ldbs + i)
            M = tl.load(m_array + i)
            N = tl.load(n_array + i)
            C = tl.load(c_ptrs + i).to(tl.pointer_type(DTYPE))
            ldc = tl.load(ldcs + i)
            D = tl.load(d_ptrs + i).to(tl.pointer_type(DTYPE))
            ldd = tl.load(ldds + i)
            alpha = tl.load(array_alpha + i)
            beta = tl.load(array_beta + i)
            # save new pid
            new_pid = pid

        pid -= b_size

    pid = new_pid
    # matrix multiplication
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_m
    group_id = pid // width
    group_size = min(grid_n - group_id * GROUP_M, GROUP_M)
    pid_n = group_id * GROUP_M + (pid % group_size)
    pid_m = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    if TRANS_A == 1:
        A = A + (ram[None, :] * lda + rk[:, None])  # KxM
    else:
        A = A + (ram[None, :] + rk[:, None] * lda)  # KxM

    if TRANS_B == 1:
        B = B + (rk[None, :] * ldb + rbn[:, None])  # NxK
    else:
        B = B + (rk[None, :] + rbn[:, None] * ldb)  # NxK

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=ACC_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            if TRANS_A == 1:
                a = tl.load(A, mask=rk[:, None] < k_remaining, other=0.)
            else:
                a = tl.load(A, mask=rk[:, None] < k_remaining, other=0.)

            if TRANS_B == 1:
                b = tl.load(B, mask=rk[None, :] < k_remaining, other=0.)
            else:
                b = tl.load(B, mask=rk[None, :] < k_remaining, other=0.)

        # do compute
        acc += tl.dot(b, a)

        if TRANS_A == 1:
            A += BLOCK_K
        else:
            A += BLOCK_K * lda

        if TRANS_B == 1:
            B += BLOCK_K * ldb
        else:
            B += BLOCK_K

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[None, :] + rn[:, None] * ldc)
    mask = (rm < M)[None, :] & (rn < N)[:, None]
    c = tl.load(C, mask=mask)

    # compute alpha * AB + beta * C
    acc = acc * alpha + beta * c

    acc = acc.to(D.dtype.element_ty)
    D = D + (rm[None, :] + rn[:, None] * ldd)
    tl.store(D, acc, mask=mask)


def triton_grouped_gemm(array_alpha, array_a, array_b, array_beta, array_c):
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
        4. tensor a,b,c are all row-major. a shape is MxK, b shape is KxN, c shape is MxN

    """
    a = array_a[0]
    device = a.device
    K = a.shape[1]

    # allocates output
    a_ptrs = []
    m_sizes = []
    n_sizes = []
    ldas = []
    b_ptrs = []
    ldbs = []
    c_ptrs = []
    ldcs = []

    block_aligned = []
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    array_d = []

    trans_a = 0
    trans_b = 0

    for a,b,c in zip(array_a, array_b, array_c):
        M = a.shape[0]
        a_ptrs.append(a.data_ptr())
        m_sizes.append(M)
        ldas.append(K)
        N = b.shape[1]
        n_sizes.append(N)
        b_ptrs.append(b.data_ptr())
        ldbs.append(N)
        c_ptrs.append(c.data_ptr())
        ldcs.append(N)

        block_aligned.append(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

    # convert list into cuda tensors
    a_ptrs_tensor = torch.tensor(tuple(a_ptrs), dtype=torch.int64, device=device)
    m_sizes_tensor = torch.tensor(tuple(m_sizes), dtype=torch.int32, device=device)
    n_sizes_tensor = torch.tensor(tuple(n_sizes), dtype=torch.int32, device=device)
    alpha_tensor = torch.tensor(tuple(array_alpha), dtype=torch.float32, device=device)
    beta_tensor = torch.tensor(tuple(array_beta), dtype=torch.float32, device=device)
    ldas_tensor = torch.tensor(tuple(ldas), dtype=torch.int32, device=device)
    b_ptrs_tensor = torch.tensor(tuple(b_ptrs), dtype=torch.int64, device=device)
    ldbs_tensor = torch.tensor(tuple(ldbs), dtype=torch.int32, device=device)
    c_ptrs_tensor = torch.tensor(tuple(c_ptrs), dtype=torch.int64, device=device)
    ldcs_tensor = torch.tensor(tuple(ldcs), dtype=torch.int32, device=device)

    block_aligned_tensor = torch.tensor(tuple(block_aligned), dtype=torch.int32, device=device)

    # T(C=AB) ==> T(C) = T(B) * T(A), column-major
    # launch kernel
    #grid = lambda META: (triton.cdiv(m_sizes[0], META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    grid = (sum(block_aligned), )
    grouped_gemm_kernel[grid](block_aligned_tensor, len(array_a),
                  n_sizes_tensor, m_sizes_tensor, K,
                  alpha_tensor,
                  b_ptrs_tensor, ldbs_tensor,
                  a_ptrs_tensor, ldas_tensor,
                  beta_tensor,
                  c_ptrs_tensor, ldcs_tensor,
                  c_ptrs_tensor, ldcs_tensor,  # re-use c_ptr as output d ptr.
                  DTYPE=tl.float32 if array_a[0].dtype == torch.float32 else tl.float16,
                  ACC_DTYPE=tl.float32,
                  TRANS_A=trans_b, TRANS_B=trans_a,  # 0 for N, 1 for T
                  BLOCK_M=BLOCK_N,
                  BLOCK_N=BLOCK_M,
                  BLOCK_K=BLOCK_K,
                  GROUP_M=2,
                  EVEN_K=int(K % BLOCK_K == 0),
                  )
    return array_c

def triton_groupedgemm_wrap(list_a, b):
    """
    list_a is a list of tensor a, with shape MxK, where M may be different.
    b is a tensor, with shape KxN.
    """
    array_alpha = []
    array_a = []
    array_b = []
    array_beta = []
    array_c = []
    for a in list_a:
      array_alpha.append(1.0)
      array_a.append(a)
      array_b.append(b)
      array_beta.append(0.0)
      M,K = a.shape[0], a.shape[1]
      K1,N = b.shape[0], b.shape[1]
      assert K == K1
      c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
      array_c.append(c)

    triton_grouped_gemm(array_alpha, array_a, array_b, array_beta, array_c)

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
    print('triton: ', triton.testing.do_bench(lambda: triton_groupedgemm_wrap(A_triton, B)))

def compare(M, K, N, device, dtype):
    B = torch.randn(K, N, device=device, dtype=dtype)

    A_list = []
    for m in M:
        a = torch.randn(m, K, device=device, dtype=dtype)
        A_list.append(a)

    # compare results
    triton_res = triton_groupedgemm_wrap(A_list, B)
    torch_res = torch_grouped_gemm(A_list, B)
    for t1, t2 in zip(triton_res, torch_res):
        if torch.allclose(t1, t2):
            print('dtype: ', dtype, ' shape: ', t1.shape, ' SAME')
        else:
            diff = abs(t1 - t2)
            print('dtype: ', dtype, ' shape: ', t1.shape, ' max diff: ', diff.max(), ' rel-diff: ', (diff / t2).max())
            #print('triton: ', t1)
            #print('torch: ', t2)


if __name__ == '__main__':
    batch=16
    M_start = 64
    M_end = 4096
    M = []
    for i in range(batch):
        m = random.randint(M_start, M_end)
        M.append(m)

    print('M: ', M)
    N = 2048
    K = 1024

    device = torch.device(0)
    torch.cuda.manual_seed(42)
    dtype = torch.float16
    #dtype = torch.float32

    compare(M, K, N, device, dtype)

    test_speed(M, K, N, device, dtype)

