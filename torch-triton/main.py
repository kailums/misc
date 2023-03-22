import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

import os
import psutil
import time
import argparse
import onnx
import numpy as np

import onnxruntime as ort
from onnxruntime import OrtValue
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BertModel, BloomModel


def get_dummy_inputs(tokenizer, batch, seq_len, past_seq_len):
    promt_str = ['Hello'] * seq_len
    promt_str = [' '.join(promt_str)] * batch
    input_ids = tokenizer(promt_str, return_tensors='pt')

    inputs = {k: v for k, v in input_ids.items()}
    input_names = list(input_ids.keys())
    return inputs, input_names
   
def get_model(args, name):
    config = AutoConfig.from_pretrained(name)
    config.num_hidden_layers = 2
    model = AutoModel.from_pretrained(name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return config, model, tokenizer

def run_torch_model(args, model, inputs, device, local_rank):
    if args.fp16:
        model.half()

    model.to(device)

    if args.compile:
        model = torch.compile(model)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    # try forward
    with torch.no_grad():
        output = model(**inputs)
        output = output.last_hidden_state
    print('output: ', output.shape, ' dtype:', output.dtype, ' dev:', output.device)

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        with torch.autograd.profiler.emit_nvtx(args.profile):
            with torch.no_grad():
                output = model(**inputs)
                output = output.last_hidden_state

        if i % interval == 0:
            cost_time = time.time() - end
            print(f'[torch] iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()
    return output

def main(args):
    local_rank = 1
    device=torch.device(local_rank)
    batch=args.batch
    seq_len=args.seq_len
    model_name = f'{args.model}'

    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)

    config, model, tokenizer = get_model(args, model_name)

    if model is not None:
        model.eval()
        model.requires_grad_(False)

    inputs, input_names = get_dummy_inputs(tokenizer, batch, seq_len, 0)

    output = run_torch_model(args, model, inputs, device, local_rank)


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--no-torch-infer', action='store_true', default=False)
    parser.add_argument('--no-ort-infer', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)
    parser.add_argument('--compile', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
