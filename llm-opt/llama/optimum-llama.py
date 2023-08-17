import torch
import os
import json
import psutil
import time
import argparse
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
import onnx
import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions


def export_model(args):
    model_path = args.model
    save_dir = args.save_dir

    # set use_merged = True to combine decoder_model.onnx and decoder_with_past_model.onnx into one model
    model = ORTModelForCausalLM.from_pretrained(model_path,export=True, use_merged=args.merge)
    model.save_pretrained(save_dir)

def convert_to_fp16(args):
    model_path = args.model
    save_dir = args.save_dir

    config = LlamaConfig.from_pretrained(model_path)
    # use ort optimizer to convert model to float16
    model_type = 'gpt2'
    opt_option=FusionOptions(model_type)
    opt_option.enable_attention=True
    opt_option.enable_flash_attention=True

    def convert(model_name, out_model_name):
        optimizer = optimize_by_fusion(
                onnx.load(model_name), 
                model_type=model_type,
                num_heads=config.num_attention_heads,
                hidden_size=config.hidden_size,
                optimization_options=opt_option
            )
        optimizer.convert_float_to_float16(use_symbolic_shape_infer=True, keep_io_types=False)

        optimizer.save_model_to_file(out_model_name, use_external_data_format=True)

    if args.merge:
        decoder_model = f'{save_dir}/decoder_model_merged.onnx'
        opt_out_file = f'{save_dir}/decoder_model_merged_fp16.onnx'
        convert(decoder_model, opt_out_file)
    else:
        decoder_model = f'{save_dir}/decoder_model.onnx'
        opt_out_file = f'{save_dir}/decoder_model_fp16.onnx'
        convert(decoder_model, opt_out_file)

        decoder_past_model = f'{save_dir}/decoder_with_past_model.onnx'
        past_out_file = f'{save_dir}/decoder_with_past_model_fp16.onnx'
        convert(decoder_past_model, past_out_file)

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

    if args.profile:
        so.enable_profiling = args.profile
        so.profile_file_prefix=f'ort-profile-rank{local_rank}'

    provider_opt = {'device_id': local_rank, 'tunable_op_enable': args.tune, 'tunable_op_tuning_enable': args.tune}

    return so, provider_opt

def setup_ort_model(args, local_rank):
    model_path = args.model
    save_dir = args.save_dir

    config = LlamaConfig.from_pretrained(model_path)
    sess_opt, provider_opt = setup_session_option(args, local_rank)
    if args.merge:
        decoder_model = f'{save_dir}/decoder_model_merged.onnx'
        decoder_past_model = None
    else:
        if config.torch_dtype != torch.float16:
            decoder_model = f'{save_dir}/decoder_model_fp16.onnx'
            decoder_past_model = f'{save_dir}/decoder_with_past_model_fp16.onnx'
        else:
            decoder_model = f'{save_dir}/decoder_model.onnx'
            decoder_past_model = f'{save_dir}/decoder_with_past_model.onnx'
    
    session, past_session = ORTModelForCausalLM.load_model(decoder_model, decoder_past_model, provider='ROCMExecutionProvider', session_options=sess_opt, provider_options=provider_opt)
    
    model = ORTModelForCausalLM(session, config, [decoder_model, decoder_past_model], past_session, use_cache=True, use_io_binding=True, model_save_dir=save_dir)

    return model


def setup_torch_model(args, local_rank):
    config = LlamaConfig.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=config.torch_dtype, config=config)
    model.to(torch.device(local_rank))
    model.eval()
    model.requires_grad_(False)
    return model

def run_generate(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    prompt='Q: What is the largest animal?\nA:'
    
    inputs = tokenizer(prompt, return_tensors='pt')
    
    if args.ort:
        ort_model = setup_ort_model(args, local_rank)
        input_ids = inputs.input_ids.to(ort_model.device)
        outputs = ort_model.generate(input_ids=input_ids, max_new_tokens=32)
        print('input ids size: ', inputs.input_ids.shape, ' value: ', inputs.input_ids)
        print('output size: ', outputs[0].shape, ' value: ', outputs[0])
    
        response = tokenizer.decode(outputs[0][1:], skip_special_token=True)
        print('[ORT] Response:', response)

    if args.torch:
        torch_model = setup_torch_model(args, local_rank)
        input_ids = inputs.input_ids.to(torch_model.device)
        outputs = torch_model.generate(input_ids=input_ids, max_new_tokens=32)
        response = tokenizer.decode(outputs[0][1:], skip_special_token=True)
        print('output size: ', outputs[0].dtype, ' value: ', outputs[0])
        print('[Torch] Response:', response)

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
    print(f'[{name}]: prmpot_len: {input_ids.shape[1]}, generate_len: {gen_len}, cost: {cost / args.loop_cnt}s')

def run_benchmark(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    if args.ort:
        ort_model = setup_ort_model(args, local_rank)
    if args.torch:
        torch_model = setup_torch_model(args, local_rank)

    batch=1
    #prompt_len = ['32', '64', '128', '256', '512', '1024']
    prompt_len = ['32']
    #generate_len = [1, 129]
    generate_len = [1]

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
            if args.ort:
                input_ids = inputs.input_ids.to(ort_model.device)
                func_benchmark(args, "ORT", ort_model, input_ids, gen_len)
            if args.torch:
                input_ids = inputs.input_ids.to(torch_model.device)
                func_benchmark(args, "Torch", torch_model, input_ids, gen_len)


def main(args):
    local_rank = 8
    if args.export:
        export_model(args)

    if args.convert_fp16:
        convert_to_fp16(args)

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
