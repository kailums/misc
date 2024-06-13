from transformers import LlamaConfig, LlamaTokenizer
from transformers import LlamaForCausalLM
#from models.modeling_llama import LlamaForCausalLM
import torch
import json
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

def main():
    init_dist()
    rank = get_rank()

    model_id = 'meta-llama/Llama-2-7b-hf'
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    config = LlamaConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2

    model = LlamaForCausalLM.from_pretrained(model_id, config=config, torch_dtype=config.torch_dtype)

    prompt = 'Once upon a time,'
    with open('prompt.json') as fp:
        prompt_pool = json.load(fp)

    prompt = prompt_pool['32']
    inputs = tokenizer(prompt, return_tensors='pt')

    device = torch.device(rank)
    model.to(device)
    model.generate = torch.compile(model.generate)
    outputs = model.generate(input_ids=inputs.input_ids.to(device), max_new_tokens=3)

    response = tokenizer.decode(outputs[0][1:], skip_special_token=True)

    print('response: ', response)


if __name__ == '__main__':
    main()
