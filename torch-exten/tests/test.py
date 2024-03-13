import torch

from te import matmul_cublasLt
import torch.nn.functional as F


def benchmark(f, m, n, k, device='cuda:0', dtype=torch.float16, warmup=1, iters=5):
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(f'm-{m}-n-{n}-k-{k}')
    for _ in range(warmup):
      c = f(a, b)
    start.record()
    for _ in range(iters):
        c = f(a, b)
    end.record()
    torch.cuda.nvtx.range_pop()
    #print('m: ', m, ' c: ', c[0, :5])

    torch.cuda.synchronize(device)
    elapsed_time = start.elapsed_time(end)

    return elapsed_time / iters

def test_benchmark(device):
    N, K = 4096, 4096
    max_m = 1024
    #f = matmul_cublasLt
    #f = torch.matmul
    f = F.linear
    cost_times = {}
    for m in range(1, max_m+1):
      cost_time = benchmark(f, m, N, K, warmup=0, iters=2, device=device, dtype=torch.float16)
      cost_times[m] = cost_time

    with open('benchmark-torch-matmul.txt', 'w') as fp:
        for m, c in cost_times.items():
          fp.write(f'M: {m} cost: {c} ms\n')

def test_matmul_cublasLt():
    m, n, k = 1024, 1024, 1024
    dtype = torch.float16
    a = torch.randn(m, k, dtype=dtype, device='cuda')
    b = torch.randn(k, n, dtype=dtype, device='cuda')
    torch.cuda.nvtx.range_push('cublaslt')
    c = matmul_cublasLt(a, b)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push('matmul')
    c2 = torch.matmul(a, b)
    torch.cuda.nvtx.range_pop()
    if not torch.allclose(c, c2, atol=1e-5):
        print('diff: ', (c - c2).abs().max())
        print('c: ', c[0, :5])
        print('c2: ', c2[0, :5])
    else:
        print('test_matmul_cublasLt passed')

if __name__ == '__main__':
    device_id = 0
    device = torch.device(device_id)
    torch.cuda.set_device(device)
    test_benchmark(device)
    #test_matmul_cublasLt()
