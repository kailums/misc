import argparse
import csv
import torch
import triton
import triton.language as tl
from itertools import product


def early_config_prune(configs, args):
    pruned_config = []
    for config in configs:
        kw = config.kwargs
        bl_size = kw['BLOCK_SIZE']
        hw_size = kw['HW_SIZE']
        img_size = args['img_size']
        c_per_group = args['c_per_group']
        if c_per_group < bl_size and hw_size < img_size and img_size % hw_size == 0:
            pruned_config.append(config)
    return pruned_config

def gen_autotune_config():
    block_size = [16, 32, 64, 128, 256, 512]
    hw_size = [8, 16, 32, 64, 128, 256, 512]
    warps = [1, 2, 4, 8]
    configs = []
    for b, hw, w in product(block_size, hw_size, warps):
        configs.append(triton.Config({'BLOCK_SIZE': b, 'HW_SIZE': hw}, num_stages=3, num_warps=w))
    return configs


@triton.autotune(
    configs=gen_autotune_config(),
    key=['img_size', 'c', 'c_per_group'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': None,
        'top_k': None,
    }
)
@triton.heuristics(
    {
        'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['c_per_group']),
    }
)
@triton.jit
def group_norm_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    img_size,
    c,
    c_per_group,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HW_SIZE: tl.constexpr,
    ACTIVATION_SWISH: tl.constexpr,
):
    row_x = tl.program_id(0)
    row_y = tl.program_id(1)
    stride = img_size * c
    input_ptr += row_x * stride + row_y * c_per_group
    output_ptr += row_x * stride + row_y * c_per_group
    gamma_ptr += row_y * c_per_group
    beta_ptr += row_y * c_per_group

    cols = tl.arange(0, BLOCK_SIZE)
    hw = tl.arange(0, HW_SIZE)
    offsets = hw[:, None]*c + cols[None, :]
    mask = (hw < img_size)[:, None] & (cols < c_per_group)[None,:]

    # Calculate mean and variance
    _sum = tl.zeros([HW_SIZE, BLOCK_SIZE], dtype=tl.float32)
    _square_sum = tl.zeros([HW_SIZE, BLOCK_SIZE], dtype=tl.float32)
    x_ptr = input_ptr 
    for i in range(tl.cdiv(img_size, HW_SIZE)):
        a = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        _sum += a
        _square_sum += a * a
        x_ptr += c * HW_SIZE   # step to next

    _sum = tl.sum(_sum, axis=0)
    _square_sum =  tl.sum(_square_sum, axis=0)
    group_mean = tl.sum(_sum, axis=0) / (img_size * c_per_group)
    group_var = tl.sum(_square_sum, axis=0) / (img_size * c_per_group) - group_mean * group_mean

    rstd = 1 / tl.sqrt(group_var + eps)

    # Normalize and apply linear transformation
    gamma = tl.load(gamma_ptr + cols, mask=cols < c_per_group).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=cols < c_per_group).to(tl.float32)
    
    for i in range(tl.cdiv(img_size, HW_SIZE)):
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - group_mean) * rstd
        y = x_hat * gamma + beta
        if ACTIVATION_SWISH:
            y *= tl.sigmoid(y)
        tl.store(output_ptr + offsets, y, mask=mask)
        input_ptr += c * HW_SIZE
        output_ptr += c * HW_SIZE


def group_norm(x, gamma, beta, eps, groups):
    n, h, w, c = x.shape
    c_per_group = c // groups
    BLOCK_SIZE = triton.next_power_of_2(c_per_group)
    num_warps = 2
    act = 0

    y = torch.empty_like(x)

    img_size = h * w
    group_norm_kernel[(n, groups)](
        x, y, gamma, beta, img_size, c, c_per_group, eps, 
        ACTIVATION_SWISH=act,
        #BLOCK_SIZE=BLOCK_SIZE, HW_SIZE=32,
        #num_warps=num_warps
    )
    #print(group_norm_kernel.best_config)
    return y


def get_input_shapes():
    return (
        (2, 64, 64, 32, 10),
        (2, 32, 32, 32, 20),
        (2, 16, 16, 32, 40),
        (2, 64, 64, 32, 256),
        (2, 16, 16, 32, 80),
        (2, 32, 32, 32, 40),
        (2, 32, 32, 32, 60),
        (2, 8, 8, 32, 40),
        (2, 64, 64, 32, 30),
        (2, 32, 32, 32, 30),
        (2, 32, 32, 32, 10),
        (2, 16, 16, 32, 20),
        (2, 16, 16, 32, 60),
        (2, 8, 8, 32, 80),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("-c", "--csv", type=str, default="group_norm_triton")

    return parser.parse_args()

def torch_group_norm(x, g, w, b, eps):
    x = torch.permute(x, (0, 3, 1, 2))
    x = torch.nn.functional.group_norm(x, g, w, b, eps)
    x = torch.permute(x, (0, 2, 3, 1))
    return x

def main():
    args = parse_args()
    use_fp16 = args.fp16

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = args.csv + ("_fp16" if use_fp16 else "_fp32") + f".{timestamp}.csv"

    f = open(csv_file, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["n", "h", "w", "g", "channels_per_group", "ms", "min_ms", "max_ms", "(GB/s)"])

    for n, h, w, g, channels_per_group in get_input_shapes():
        dtype = torch.float16 if use_fp16 else torch.float32
        c = g * channels_per_group
        x = torch.randn(n, h, w, c, device='cuda', dtype=dtype)
        gamma = torch.randn(c, device='cuda', dtype=dtype)
        beta = torch.randn(c, device='cuda', dtype=dtype)
        eps = 0.05

        #triton_res = group_norm(x, gamma, beta, eps, g)
        #torch_res = torch_group_norm(x, g, gamma, beta, eps)
        #if torch.allclose(triton_res, torch_res):
        #    print('SMAE')
        #else:
        #    print('triton: ', triton_res)
        #    print('torch: ', torch_res)
        #    diff = abs(torch_res - triton_res)
        #    print('NOT SAME, max diff: ', diff.max())

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: group_norm(x, gamma, beta, eps, g), quantiles=quantiles)
        # element_size returns the size of the element in bytes
        bandwidth = (x.numel() * 2 + gamma.numel() * 2) * x.element_size() / ms / 1e6
        writer.writerow([n, h, w, g, channels_per_group, ms, min_ms, max_ms, bandwidth])


if __name__ == "__main__":
    main()
