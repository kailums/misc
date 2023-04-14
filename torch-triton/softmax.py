import torch

import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

def check_size(configs, args):
    n_cols = args['n_cols']
    ret_conf = []
    for c in configs:
        if c.kwargs['BLOCK_SIZE'] >= n_cols:
            ret_conf.append(c)
    return ret_conf

def generate_configs(BLOCK_SIZE):
    configs = []
    stags = [1,2,3,4]
    warps = [2,4,8,16,32]
    for s in stags:
        for w in warps:
            configs.append(triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_stages=s, num_warps=w))
    return configs

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE':4096}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=3, num_warps=32),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=4, num_warps=16),
        triton.Config({'BLOCK_SIZE':4096}, num_stages=4, num_warps=32),
    ],
    key=['n_cols', 'n_rows'],
    prune_configs_by={'perf_model':None, 'top_k': None, 'early_config_prune': check_size}
)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, n_rows,
    BLOCK_SIZE: tl.constexpr
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)



def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel.configs=generate_configs(BLOCK_SIZE)
    softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_cols, n_rows, grid=(n_rows,))
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        n_rows,
        #num_warps=num_warps,
        #BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


#@triton.testing.perf_report(
#    triton.testing.Benchmark(
#        x_names=['N'],  # argument names to use as an x-axis for the plot
#        x_vals=[
#            128 * i for i in range(2, 100)
#        ],  # different possible values for `x_name`
#        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
#        line_vals=[
#            'triton',
#            'torch-native',
#            'torch-jit',
#        ],  # possible values for `line_arg``
#        line_names=[
#            "Triton",
#            "Torch (native)",
#            "Torch (jit)",
#        ],  # label name for the lines
#        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
#        ylabel="GB/s",  # label name for the y-axis
#        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
#        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
#    )
#)
#def benchmark(M, N, provider):
#    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
#    quantiles = [0.5, 0.2, 0.8]
#    if provider == 'torch-native':
#        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
#    if provider == 'triton':
#        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
#    if provider == 'torch-jit':
#        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
#    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
#    return gbps(ms), gbps(max_ms), gbps(min_ms)
#
#
#benchmark.run(show_plots=True, print_data=True)

def compare():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


if __name__ == '__main__':
    batches = [1, 8, 64, 1024, 4096, 65536]
    #batches = [4096, 65536]
    #N = [512, 1024, 2048, 4096]
    N = [4096]
    for b in batches:
        for n in N:
            dtype = torch.float16
            quantiles = [0.5, 0.2, 0.8]
            x = torch.randn(b, n, device='cuda', dtype=dtype)
            print('x shape: ', x.shape)
            print('torch: ', triton.testing.do_bench(lambda: torch.softmax(x, axis=-1)))
            print('triton: ', triton.testing.do_bench(lambda: softmax(x)))
            print('best: ', softmax_kernel.best_config)

