import torch

import triton
import triton.language as tl

import random
import argparse
from triton.ops import matmul
from itertools import product

import sys
import os
if os.path.exists("/ws/work/pytorch_grouped_gemm/build"):
    sys.path.append("/ws/work/pytorch_grouped_gemm/build")
    import PYTORCH_GROUPED_GEMM
else:
    PYTORCH_GROUPED_GEMM = None

def gen_tune_config():
    m_range = [16, 32, 64]
    n_range = [16, 32, 64]
    k_range = [32, 64, 128]
    stages = [1,2,3]
    warps = [4, 8, 16, 32]
    configs = []
    for m,n,k,s,w in product(m_range, n_range, k_range, stages, warps):
        configs.append(triton.Config({'BLOCK_M':m, 'BLOCK_N':n, 'BLOCK_K':k, 'GROUP_M':8}, num_stages=s, num_warps=w))

    return configs


#@triton.autotune(
#    #configs=gen_tune_config(),
#    configs=[
#       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
#    ],
#    key=['K'],
#)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def grouped_gemm_kernel(block_mids, block_offset, num_of_mids,
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
        BETA_ZERO: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    ):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # reset DTYPE, because it is tl.constexpr, can't be used in tl.pointer_type
    if DTYPE == tl.constexpr(tl.float16):
        DTYPE = tl.float16
    else:
        DTYPE = tl.float32

    while pid < num_of_mids:
        i = tl.load(block_mids + pid)
        work_offset = tl.load(block_offset + pid)

        # search for A pointer, M, C pointer of current pid
        M = tl.load(m_array + i)
        N = tl.load(n_array + i)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)

        A = tl.load(a_ptrs + i).to(tl.pointer_type(DTYPE))
        lda = tl.load(ldas + i)
        B = tl.load(b_ptrs + i).to(tl.pointer_type(DTYPE))
        ldb = tl.load(ldbs + i)
        C = tl.load(c_ptrs + i).to(tl.pointer_type(DTYPE))
        ldc = tl.load(ldcs + i)
        D = tl.load(d_ptrs + i).to(tl.pointer_type(DTYPE))
        ldd = tl.load(ldds + i)
        alpha = tl.load(array_alpha + i)
        beta = tl.load(array_beta + i)

        # matrix multiplication
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_m
        group_id = work_offset // width
        group_size = min(grid_n - group_id * GROUP_M, GROUP_M)
        pid_n = group_id * GROUP_M + (work_offset % group_size)
        pid_m = (work_offset % width) // (group_size)
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
        mask = (rm < M)[None, :] & (rn < N)[:, None]
        if BETA_ZERO:
            acc = acc * alpha
        else:
            C = C + (rm[None, :] + rn[:, None] * ldc)
            c = tl.load(C, mask=mask)
        
            # compute alpha * AB + beta * C
            acc = acc * alpha + beta * c
        
        acc = acc.to(D.dtype.element_ty)
        D = D + (rm[None, :] + rn[:, None] * ldd)
        tl.store(D, acc, mask=mask)
        pid += num_pids


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
    BLOCK_M=128
    BLOCK_N=64

    block_offset = []
    block_mids = []

    array_d = []

    trans_a = 0
    trans_b = 0

    for i, (a,b,c) in enumerate(zip(array_a, array_b, array_c)):
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

        mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        block_mids.extend([i for _ in range(mn)])
        block_offset.extend([off for off in range(mn)])

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

    block_offset_tensor = torch.tensor(tuple(block_offset), dtype=torch.int32, device=device)
    block_mids_tensor = torch.tensor(tuple(block_mids), dtype=torch.int32, device=device)

    # T(C=AB) ==> T(C) = T(B) * T(A), column-major
    # launch kernel
    def get_grid(**kwargs):
        m_sizes_tensor = kwargs['m_array']
        n_sizes_tensor = kwargs['n_array']
        BLOCK_M = kwargs['BLOCK_M']
        BLOCK_N = kwargs['BLOCK_N']

        ret = 0
        m = (m_sizes_tensor / BLOCK_M).ceil().to(torch.int32)
        n = (n_sizes_tensor / BLOCK_N).ceil().to(torch.int32)
        ret = sum(m * n)
        #ret = sum(m)
        #print('grid: ', ret)
        
        return (ret,)
    grid = lambda META: get_grid(**META)
    grouped_gemm_kernel[grid](block_mids_tensor, block_offset_tensor, len(block_mids),
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
                  BETA_ZERO=1,  # beta is zero
                  BLOCK_M=BLOCK_N,
                  BLOCK_N=BLOCK_M,
                  BLOCK_K=128,
                  GROUP_M=8,
                  num_warps=8,
                  #EVEN_K=int(K % BLOCK_K == 0),
                  )
    return array_c

def triton_groupedgemm_wrap(list_a, list_b):
    """
    list_a is a list of tensor a, with shape MxK, where M may be different.
    b is a tensor, with shape KxN.
    """
    array_alpha = []
    array_a = []
    array_b = []
    array_beta = []
    array_c = []
    for a,b in zip(list_a, list_b):
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

def torch_grouped_gemm(list_a, list_b):
    list_c = []
    #if PYTORCH_GROUPED_GEMM is not None and list_a[0].dtype != torch.float16:
    if PYTORCH_GROUPED_GEMM is not None:
        for a, b in zip(list_a, list_b):
            M, K = a.shape[0], a.shape[1]
            N = b.shape[1]
            c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
            list_c.append(c)

        PYTORCH_GROUPED_GEMM.GroupedGEMM(list_a, list_b, list_c, list_c, 1.0, 0.0)
    else:
        for a, b in zip(list_a, list_b):
            c = torch.matmul(a, b)
            list_c.append(c)
    
    return list_c

def triton_matmul(list_a, list_b):
    list_c = []
    for a, b in zip(list_a, list_b):
        c = matmul(a, b)
        list_c.append(c)
    return list_c
 

def test_speed(mnk_array, device, dtype):
    a_list = []
    b_list = []
    max_m, max_n, max_k = 0, 0, 0
    # generate input data for triton
    for (m, n, k) in mnk_array:
        a_list.append(torch.randn(m, k, device=device, dtype=dtype))
        b_list.append(torch.randn(k, n, device=device, dtype=dtype))

    print('torch: ', triton.testing.do_bench(lambda: torch_grouped_gemm(a_list, b_list)))
    #print('triton-matmul: ', triton.testing.do_bench(lambda: triton_matmul(a_list, b_list)))
    print('triton-groupedgemm: ', triton.testing.do_bench(lambda: triton_groupedgemm_wrap(a_list, b_list)))

def compare_result(mnk_array, device, dtype):
    a_list = []
    b_list = []
    # generate input data for triton
    for (m, n, k) in mnk_array:
        a_list.append(torch.randn(m, k, device=device, dtype=dtype))
        b_list.append(torch.randn(k, n, device=device, dtype=dtype))

    # compare results
    triton_res = triton_groupedgemm_wrap(a_list, b_list)
    torch_res = torch_grouped_gemm(a_list, b_list)
    for t1, t2 in zip(triton_res, torch_res):
        if torch.allclose(t1, t2):
            print('dtype: ', dtype, ' shape: ', t1.shape, ' SAME')
        else:
            diff = abs(t1 - t2)
            print('dtype: ', dtype, ' shape: ', t1.shape, ' max diff: ', diff.max(), ' rel-diff: ', (diff / t2).max())
            #print('triton: ', t1)
            #print('torch: ', t2)

def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--compare', action='store_true', default=False)
    parser.add_argument('--speed', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arges()
    batch=16
    N_start = 100
    N_end = 200
    N = []
    M = 768
    K = 4608
    row1 = []
    random.seed(42)
    for i in range(batch):
        n = random.randint(N_start, N_end) // 8 * 8  # align to 8
        row1.append([M, n, K])

    row2 = [[768,1,4608], [768,1,4608]]
    row3 = [[4608,1,384], [4608,1,384]]
    row6 = [ [ 768, 2, 4608], [ 768, 1, 4608], [ 768, 1, 4608], [ 768, 1, 4608], [ 768, 1, 4608], [ 768, 1, 4608], [ 768, 3, 4608], [ 768, 4, 4608], [ 768, 3, 4608], [ 768, 5, 4608], [ 768, 2, 4608], [ 768, 4, 4608], [ 768, 2, 4608], [ 768, 1, 4608], [ 768, 1, 4608]]
    row7 = [[4608, 2, 384], [4608, 1, 384], [4608, 1, 384], [4608, 1, 384], [4608, 1, 384], [4608, 1, 384], [4608, 3, 384], [4608, 4, 384], [4608, 3, 384], [4608, 5, 384], [4608, 2, 384], [4608, 4, 384], [4608, 2, 384], [4608, 1, 384], [4608, 1, 384]]
    row8 = [[768, 167, 4608], [768, 183, 4608], [768, 177, 4608], [768, 181, 4608], [768, 153, 4608], [768, 139, 4608], [768, 156, 4608], [768, 173, 4608], [768, 163, 4608], [768, 150, 4608], [768, 204, 4608], [768, 184, 4608], [768, 168, 4608], [768, 156, 4608], [768, 168, 4608], [768, 148, 4608]]
    row9 = [[4608, 167, 384], [4608, 183, 384], [4608, 177, 384], [4608, 181, 384], [4608, 153, 384], [4608, 139, 384], [4608, 156, 384], [4608, 173, 384], [4608, 163, 384], [4608, 150, 384], [4608, 204, 384], [4608, 184, 384], [4608, 168, 384], [4608, 156, 384], [4608, 168, 384], [4608, 148, 384]]

    device = torch.device(0)
    #device = torch.device(3)
    torch.cuda.manual_seed(42)
    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16

    #for i, mnk in enumerate([row2, row3, row6, row7, row8, row9]):
    for i, mnk in enumerate([row1]):
        print('test row ', mnk)
        if args.compare:
            compare_result(mnk, device, dtype)
        if args.speed:
            test_speed(mnk, device, dtype)

    #print('best: ', grouped_gemm_kernel.best_config)
