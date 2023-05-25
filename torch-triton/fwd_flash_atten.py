import pytest
import torch

import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)  # (m,)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)  # (m,)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])  # (m,k)
        l_curr = tl.sum(p, 1) + l_prev   # (m)
        # rescale operands of matmuls
        l_rcp = 1. / l_curr    # (m,)
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]

        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

def flash_att_fwd(q, k, v):
    BLOCK = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    sm_scale = 1 / math.sqrt(q.size(-1))

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        L, m,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk, num_warps=num_warps,
        num_stages=2,
    )
    return o

def torch_fl_attention(q, k, v):
    atten = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    atten = torch.softmax(atten.float(), dim=-1).half()
    return atten @ v

if __name__ == '__main__':
    shape = (8, 16, 64, 16)
    device = torch.device(0)
    requires_grad = False
    dtype = torch.float16
    q = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)

    print('forward')
    #triton_out = flash_att_fwd(q, k.transpose(-2, -1), v)
    triton_out = flash_att_fwd(q, k, v)
    print('out: ', triton_out.shape)

    #with torch.backends.cuda.enable_flash_sdp(True):
    #with torch.backends.cuda.sdp_kernel(enable_math=False):
    #torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch_out = torch_fl_attention(q, k, v)
    assert torch.allclose(triton_out, torch_out, rtol=0.1, atol=0.1), (triton_out, torch_out)

    #print('torch: ', triton.testing.do_bench(lambda: torch_fl_attention(q, k, v)))
    #print('triton: ', triton.testing.do_bench(lambda: flash_att_fwd(q, k.transpose(-2, -1), v)))


