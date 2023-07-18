import torch
from torch import nn
import torch.distributed as dist

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

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_rank():
    return 0

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

    return so

def run_onnxruntime(args, model_file, inputs):
    local_rank = get_rank()
    model_file = f'{args.save_dir}/{model_file}'
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank, 'tunable_op_enable': args.tune, 'tunable_op_tuning_enable': args.tune})])
    io_binding = sess.io_binding()

    # bind inputs by using OrtValue
    input_names = sess.get_inputs()
    for k in input_names:
        np_data = inputs[k.name].cpu().numpy()
        x = OrtValue.ortvalue_from_numpy(np_data, 'cuda', local_rank)
        io_binding.bind_ortvalue_input(k.name, x)
    # bind outputs
    outputs = sess.get_outputs()
    for out in outputs:
        io_binding.bind_output(out.name, 'cuda', local_rank)

    sess.run_with_iobinding(io_binding)

    output = io_binding.copy_outputs_to_cpu()

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        sess.run_with_iobinding(io_binding)

        if i % interval == 0:
            cost_time = time.time() - end
            print(f'iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()

    return output

def get_dummy_inputs(model_name, batch, seq_len, past_seq_len, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    promt_str = ['hello'] * seq_len
    promt_str = ' '.join(promt_str)
    promt_str = [promt_str] * batch
    input_ids = tokenizer(promt_str, return_tensors='pt')
    att_mask = input_ids['attention_mask'].to(device)
    input_ids = input_ids['input_ids'].to(device)
    print('ids shape: ', input_ids.shape, ' mask shape: ', att_mask.shape)

    inputs = (input_ids, att_mask)
    input_names = ['input_ids', 'attention_mask']
    return {k: v for k, v in zip(input_names, inputs)}

def get_model(args, name):
    config = AutoConfig.from_pretrained(name)
    config.n_layer=2
    config.alibi=True
    #model = LlamaForCausalLM(config)
    #model = LlamaForCausalLM.from_pretrained(name, torch_dtype=config.torch_dtype, config=config)
    model = AutoModelForCausalLM.from_pretrained(name, config=config)
    return config, model


def run_torch_model(args, model, inputs, device):
    if args.fp16:
        model.half()

    model.to(device)

    if args.compile:
        model = torch.compile(model)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    # try forward
    with torch.no_grad():
        output = model(**inputs)
        output = output.logits
    print('output: ', output.shape, ' dtype:', output.dtype, ' dev:', output.device)

    local_rank = get_rank()

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        with torch.autograd.profiler.emit_nvtx(args.profile):
            with torch.no_grad():
                output = model(**inputs)
                output = output.logits

        if i % interval == 0:
            cost_time = time.time() - end
            print(f'[torch] iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()
    return output


def export_model(args, model, config, local_rank, inputs, input_names, output_names, model_out_file, tmp_out_file, opt_out_file): 
    # sync all process
    print('start to export model')

    print(f'rank: {local_rank} start to export onnx')
    torch.onnx.export(
            model,
            f=tmp_out_file,
            args=inputs,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            custom_opsets={'com.microsoft':1},
            export_params=True, # need these config to keep result same as torch
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(tmp_out_file)
    onnx.save(onnx_model, model_out_file, save_as_external_data=True, location=f'{model_out_file}.data')
    print(f'rank: {local_rank} export to onnx done.')

    ## use fusion to optimize model
    if args.opt_export:
        model_type = 'bert'
        opt_option=FusionOptions(model_type)
        opt_option.enable_attention=False
        opt_option.enable_flash_attention=False
        optimizer = optimize_by_fusion(
                onnx.load(model_out_file), 
                model_type=model_type,
                num_heads=config.num_attention_heads,
                hidden_size=config.hidden_size,
                optimization_options=opt_option
            )
        if args.fp16:
            optimizer.convert_float_to_float16(use_symbolic_shape_infer=True, keep_io_types=False)

        optimizer.save_model_to_file(opt_out_file, use_external_data_format=True)
        print(f'rank: {local_rank}, save optimized onnx done.')

def compare(args, nparray1, nparray2):
    def l2_compare(a1, a2, tol=2**-7):
        a1 = a1.astype(np.float32)
        a2 = a2.astype(np.float32)
        diff = np.abs(a1 - a2)
        avgval = np.average(abs(a1))
        l2_err = 0.0 if avgval == 0 else np.sqrt(np.square(diff).sum()) / np.sqrt(np.square(a1).sum())
        if l2_err > tol:
            print(f"fp16 compare: mismatch! l2_err: {l2_err}")
        else:
            print(f"fp16 compare: passed! l2_err: {l2_err} avg: {avgval}")

    if args.fp16:
        return l2_compare(nparray1, nparray2)

    if np.allclose(nparray1, nparray2, atol=1e-5):
        print('result SAME.')
    else:
        diff = abs(nparray1 - nparray2)
        rel_diff = abs(diff / nparray1)
        print(f'not SAME, max diff: {diff.max()}, rel-diff: {rel_diff.max()}')


def main(args):
    batch=args.batch
    seq_len=args.seq_len
    model_name = f'{args.model}'

    local_rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)

    config, model = get_model(args, model_name)

    if model is not None:
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        # if args.fp16:
        #     model.half()

    inputs = get_dummy_inputs(model_name, batch, seq_len, None, device)
    input_names = list(inputs.keys())
    output_names = ['output']

    # export to onnx
    model_out_file = f'rank-{local_rank}-{args.output}'
    tmp_file = f'tmp-{model_out_file}'
    opt_out_file = f'opt-{model_out_file}'
    if args.export:
        export_model(args, model, config, local_rank, inputs, input_names, output_names, model_out_file, tmp_file, opt_out_file)

    if not args.no_torch_infer:
        output = run_torch_model(args, model, inputs, device)

    if not args.no_ort_infer:
        #model_out_file = './llama-7b-hf/decoder_model.onnx'
        ort_out = run_onnxruntime(args, opt_out_file if args.opt_export else model_out_file, inputs)

    if args.no_torch_infer or args.no_ort_infer:
        return

    o1 = output.cpu().numpy()

    compare(args, o1, ort_out[0])


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
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--opt-export', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
