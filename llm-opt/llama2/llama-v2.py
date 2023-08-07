import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

import os
import psutil
import time
import argparse
import onnx
import json
import numpy as np
from mpi4py import MPI

import onnxruntime as ort
from onnxruntime import OrtValue
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions

from transformers import LlamaConfig, LlamaTokenizer
#from transformers import LlamaForCausalLM
from models.modeling_llama import LlamaForCausalLM, LlamaAttention
from models.parallel_layers import TensorParallelColumnLinear, TensorParallelRowLinear

def init_dist():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK',0))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:9876', world_size=world_size, rank=rank)
    device = torch.device(local_rank)
    return device

def get_rank():
    comm = MPI.COMM_WORLD
    return comm.Get_rank()

def get_size():
    comm = MPI.COMM_WORLD
    return comm.Get_size()

def get_process_group():
    return _get_default_group()

def barrier():
    comm = MPI.COMM_WORLD
    comm.Barrier()

def print_out(*args):
    if get_rank() == 0:
        print(*args)

def broadcast(data):
    comm = MPI.COMM_WORLD
    comm.broadcast(data, root=0)

def get_input_prompt():
    rank = get_rank()
    if rank == 0:
        prompt = input('human')
    else:
        prompt = ''
    broadcast(prompt, root=0)
    return prompt

def _split_model(model):
    if isinstance(model, TensorParallelColumnLinear) or isinstance(model, TensorParallelRowLinear) or isinstance(model, LlamaAttention):
        model.parallel_split()
    for _, m in model._modules.items():
        _split_model(m)

   
def setup_torch_model(args, local_rank):
    import torch.nn.init
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x

    barrier()
    world_size = get_size()
    for i in range(world_size):
        if i == local_rank:
            config = LlamaConfig.from_pretrained(args.model)
            #config.num_hidden_layers=2
            model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=config.torch_dtype, config=config)
            _split_model(model)
            #model = LlamaForCausalLM.from_pretrained(args.model, config=config)
            if local_rank >= 0:
                model.to(torch.device(local_rank))
            model.eval()
            model.requires_grad_(False)
        barrier()
    return model

def run_generate(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    #prompt='Q: What is the largest animal?\nA:'
    prompt='Once upon a time,'
    
    inputs = tokenizer(prompt, return_tensors='pt')
    
    if args.torch:
        torch_model = setup_torch_model(args, local_rank)
        input_ids = inputs.input_ids.to(torch_model.device)
        outputs = torch_model.generate(input_ids=input_ids, max_new_tokens=32)
        response = tokenizer.decode(outputs[0][1:], skip_special_token=True)
        print_out('[Torch] Response:', response)

def func_benchmark(args, name, ort_model, input_ids, gen_len):
    for _ in range(args.warm):
        torch.cuda.nvtx.range_push('generate')
        outputs = ort_model.generate(input_ids = input_ids, max_new_tokens=gen_len)
        torch.cuda.nvtx.range_pop()

    start = time.time()
    for _ in range(args.loop_cnt):
        torch.cuda.nvtx.range_push('generate')
        outputs = ort_model.generate(input_ids = input_ids, max_new_tokens=gen_len)
        torch.cuda.nvtx.range_pop()
    cost = time.time() - start
    print_out(f'[{name}]: prmpot_len: {input_ids.shape[1]}, generate_len: {gen_len}, cost: {cost / args.loop_cnt}s')

def run_benchmark(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    if args.torch:
        torch_model = setup_torch_model(args, local_rank)

    batch=1
    prompt_len = ['32', '64', '128', '256', '512', '1024']
    #prompt_len = ['32']
    generate_len = [1, 129]
    #generate_len = [1]

    for p_len in prompt_len:
        for gen_len in generate_len:
            # generate input prompt
            with open('prompt.json') as fp:
                prompt_pool = json.load(fp);
            if p_len in prompt_pool:
                prompt = prompt_pool[p_len]
            else:
                prompt = ['Hello'] * (int(p_len) - 1)
                prompt = ' '.join(prompt)

            inputs = tokenizer(prompt, return_tensors='pt')
            
            if args.torch:
                input_ids = inputs.input_ids.to(torch_model.device)
                func_benchmark(args, "Torch", torch_model, input_ids, gen_len)


def main(args):
    device = init_dist()
    local_rank = get_rank()

    if args.generate:
        run_generate(args, local_rank)

    if args.benchmark:
        run_benchmark(args, local_rank)

def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model', type=str)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--warm', type=int, default=5)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--merge', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--opt-export', action='store_true', default=False)
    parser.add_argument('--ort', action='store_true', default=False)
    parser.add_argument('--torch', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--convert-fp16', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
