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
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model import OnnxModel  # noqa: E402

from transformers import LlamaConfig, LlamaTokenizer
#from transformers import LlamaForCausalLM
from models.modeling_llama import LlamaForCausalLM

from export_to_onnx import run_torchscript_export
from ort_llama import OrtModelForLlamaCausalLM
from chatcli import chat_loop, generate_stream, generate, DistChatIO


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

def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    if args.logging:
        so.log_severity_level = 0
        ort.set_default_logger_severity(0)  # open log

    if args.ort_opt:
        so.optimized_model_filepath = f'ort-opted-rank-{local_rank}-{args.output}'

    if args.profile and local_rank == 0:
        so.enable_profiling = args.profile
        so.profile_file_prefix=f'ort-profile-rank-{local_rank}'

    provider_opt = {'device_id': local_rank, 'tunable_op_enable': args.tune, 'tunable_op_tuning_enable': args.tune}

    return so, provider_opt

def setup_ort_model(args, rank):
    config = LlamaConfig.from_pretrained(args.model)
    #config.num_hidden_layers = 2
    decoder_model = f"{args.output_name}_rank-{rank}_decoder_model_fp32.onnx"
    decoder_past_model = f"{args.output_name}_rank-{rank}_decoder_with_past_model_fp32.onnx"
    sess_opt, provider_opt = setup_session_option(args, rank)

    model = OrtModelForLlamaCausalLM(args, decoder_model, decoder_past_model, rank, sess_opt, provider_opt, config=config)
    model.to(torch.device(rank))

    return model


def setup_torch_model(args, use_cuda=True):
    import torch.nn.init
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x

    barrier()
    world_size = get_size()
    rank = get_rank()
    for i in range(world_size):
        if i == rank:
            config = LlamaConfig.from_pretrained(args.model)
            #config.num_hidden_layers=2
            model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=config.torch_dtype, config=config)
            model.parallel_model()
            if use_cuda:
                model.to(torch.device(rank))
            model.eval()
            model.requires_grad_(False)
            if args.compile:
                model.generate = torch.compile(model.generate, backend="inductor")
        barrier()
    return model

def optimize_transformer(args, model_file, opt_out_file, num_heads, hidden_size):
    #model_type = 'bert'
    model_type='t5'
    opt_option=FusionOptions(model_type)
    opt_option.enable_attention=False
    opt_option.enable_flash_attention=False
    optimizer = optimize_by_fusion(
            onnx.load(model_file), 
            model_type=model_type,
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=opt_option
        )
    if args.convert_fp16:
        optimizer.convert_float_to_float16(use_symbolic_shape_infer=True, keep_io_types=False)
    optimizer.save_model_to_file(opt_out_file, use_external_data_format=True)


def export_model(args):
    rank = get_rank()
    config = LlamaConfig.from_pretrained(args.model)
    #config.num_hidden_layers = 2
    world_size = get_size()

    # used for attention fusion, should split by TensorParallel
    num_heads = config.num_attention_heads // world_size
    hidden_size = config.hidden_size // world_size

    model = setup_torch_model(args, use_cuda=True)
    barrier()
    for i in range(world_size):
        if i == rank:
            decoder_model_fp32, decoder_with_past_fp32 = run_torchscript_export(args, config, model, rank, world_size)
            if args.opt_export:
                decoder_model_out_path = f"{args.output_name}_rank-{rank}_decoder_model_opted.onnx"
                optimize_transformer(args, decoder_model_fp32, decoder_model_out_path, num_heads, hidden_size)

                decoder_with_past_model_out_path = f"{args.output_name}_rank-{rank}_decoder_with_past_model_opted.onnx"
                optimize_transformer(args, decoder_with_past_fp32, decoder_with_past_model_out_path, num_heads, hidden_size)
        barrier()


def run_generate(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    #prompt='Q: What is the largest animal?\nA:'
    prompt='Once upon a time,'
    #prompt='Q: there are two sets of nodes in a graph, each node in one set is connecting to all nodes in the other set, what is graph called?\nA:'

    inputs = tokenizer(prompt, return_tensors='pt')

    if args.ort:
        ort_model = setup_ort_model(args, local_rank)
        input_ids = inputs.input_ids.to(ort_model.device)
        outputs = ort_model.generate(input_ids=input_ids, max_new_tokens=128)
        print_out('input ids size: ', inputs.input_ids.shape, ' value: ', inputs.input_ids)
        print_out('output size: ', outputs[0].shape, ' value: ', outputs[0])
        print_out('ort cost: ', ort_model.cost, ' iters: ', ort_model.iters)

        response = tokenizer.decode(outputs[0][1:], skip_special_token=True)
        print_out('[ORT] Response:', response)


    if args.torch:
        torch_model = setup_torch_model(args, use_cuda=True)
        input_ids = inputs.input_ids.to(torch_model.device)
        outputs = torch_model.generate(input_ids=input_ids, max_new_tokens=128)
        response = tokenizer.decode(outputs[0][1:], skip_special_token=True)
        print_out('output size: ', outputs[0].shape, ' value: ', outputs[0])
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
        #print_out('ort cost: ', ort_model.cost, ' iters: ', ort_model.iters)
        torch.cuda.nvtx.range_pop()
    cost = time.time() - start
    print_out(f'[{name}]: prmpot_len: {input_ids.shape[1]}, generate_len: {gen_len}, cost: {cost / args.loop_cnt}s')

def run_benchmark(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    if args.ort:
        ort_model = setup_ort_model(args, local_rank)

    if args.torch:
        torch_model = setup_torch_model(args, use_cuda=True)

    batch=1
    #prompt_len = ['32', '64', '128', '256', '512', '1024']
    prompt_len = ['32']
    #prompt_len = ['2017']
    #prompt_len = ['1024']
    #generate_len = [1, 129]
    generate_len = [3]

    for p_len in prompt_len:
        for gen_len in generate_len:
            # generate input prompt
            with open('prompt.json') as fp:
                prompt_pool = json.load(fp);
            if p_len in prompt_pool:
                prompt = prompt_pool[p_len]
            else:
                prompt = ['Hello'] * int(p_len)
                prompt = ' '.join(prompt)

            inputs = tokenizer(prompt, return_tensors='pt')

            if args.ort:
                input_ids = inputs.input_ids.to(ort_model.device)
                func_benchmark(args, "ORT", ort_model, input_ids, gen_len)

            if args.torch:
                input_ids = inputs.input_ids.to(torch_model.device)
                func_benchmark(args, "Torch", torch_model, input_ids, gen_len)

def run_chat(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    if args.ort:
        model = setup_ort_model(args, local_rank)

    if args.torch:
        model = setup_torch_model(args, use_cuda=True)

    world_size = get_size()
    if world_size > 1:
        chatio = DistChatIO()
    else:
        chatio = None

    chat_loop(
        model,
        tokenizer,
        generate_stream_func = generate_stream,
        #generate_func = generate,
        max_new_tokens=512,
        chatio = chatio,
    )


def main(args):
    device = init_dist()
    local_rank = get_rank()

    if args.export:
        export_model(args)

    if args.generate:
        run_generate(args, local_rank)

    if args.benchmark:
        run_benchmark(args, local_rank)

    if args.chat:
        run_chat(args, local_rank)

def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--output_name', type=str, default='.')
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--warm', type=int, default=5)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--opt_export', action='store_true', default=False)
    parser.add_argument('--ort', action='store_true', default=False)
    parser.add_argument('--torch', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--convert_fp16', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--chat', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
