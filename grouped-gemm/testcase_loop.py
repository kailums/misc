import torch

import triton
import triton.language as tl


@triton.jit
def double_while_loop_vector_add(num_vec: tl.constexpr, vec_sizes, vec_a, vec_b, vec_res, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    i = 0
    vec_size_sum = 0
    while i < num_vec:
        v_size = tl.load(vec_sizes + i)
        vec_size_next = vec_size_sum + v_size
        while pid >= vec_size_sum and pid < vec_size_next:
            A_ptr = vec_a + vec_size_sum
            B_ptr = vec_b + vec_size_sum
            res_ptr = vec_res + vec_size_sum

            pid_offset = pid - vec_size_sum
            vec_offset = pid_offset * BLOCK

            offset = pid_offset * BLOCK + tl.arange(0, BLOCK)
            data_a = tl.load(A_ptr + offset, mask=offset < v_size)
            data_b = tl.load(B_ptr + offset, mask=offset < v_size)

            data_res = data_a + data_b

            tl.store(res_ptr + offset, data_res, mask=offset < v_size)

            pid += num_pids

        i += 1
        vec_size_sum = vec_size_next

def triton_vec_add(vec_sizes, vec_a, vec_b):
    vec_res = torch.zeros_like(vec_a)
    grid = (100,)

    vec_sizes_tensor = torch.tensor(vec_sizes, dtype=torch.int32, device=vec_a.device)

    double_while_loop_vector_add[grid](len(vec_sizes),
        vec_sizes_tensor,
        vec_a,
        vec_b,
        vec_res,
        BLOCK=64)
    return vec_res

def torch_vec_add(vec_sizes, vec_a, vec_b):
    return vec_a + vec_b

def testcase_double_while(vec_sizes):
    vec_size = sum(vec_sizes)
    device = torch.device(0)
    vec_a = torch.randn(vec_size, device=device)
    vec_b = torch.randn(vec_size, device=device)

    triton_res = triton_vec_add(vec_sizes, vec_a, vec_b)
    torch_res = torch_vec_add(vec_sizes, vec_a, vec_b)

    if torch.allclose(triton_res, torch_res):
        print('Res is SAME.')
    else:
        print('Not SAME')
        print('triton: ', triton_res)
        print('torch: ', torch_res)

if __name__ == '__main__':
    testcase_double_while([96, 24, 128])
    
