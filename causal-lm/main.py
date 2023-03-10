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
from mpi4py import MPI

import onnxruntime as ort
from onnxruntime import OrtValue
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions

from transformers import AutoConfig, AutoTokenizer, AutoModel, GenerationMixin, PreTrainedModel, GenerationConfig
from transformers import BertModel, BloomForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def init_dist(args):
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

    local_rank = 2
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:9876', world_size=world_size, rank=rank)
    device = torch.device(local_rank)
    return device

def get_rank():
    comm = MPI.COMM_WORLD
    #return comm.Get_rank()
    return 2

def get_size():
    comm = MPI.COMM_WORLD
    return comm.Get_size()

def get_process_group():
    return _get_default_group()

def barrier():
    comm = MPI.COMM_WORLD
    comm.Barrier()

def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    ##if local_rank == 0:
    #so.log_severity_level = 0
    #ort.set_default_logger_severity(0)  # open log
    if args.ort_opt:
        so.optimized_model_filepath = f'ort-opted-rank-{local_rank}-{args.output}'

    if args.profile:
        so.enable_profiling = args.profile
        so.profile_file_prefix=f'ort-profile-rank{local_rank}'

    return so

class GenerateWrapper(PreTrainedModel):
    def __init__(self, infer_func, **kwargs):
        super().__init__(**kwargs)
        self.infer_func = infer_func
        self.one = nn.Parameter(torch.tensor([0]), requires_grad=False)

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, input_ids, attention_mask, **kwargs):
        ret = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return ret

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.infer_func(input_ids, attention_mask, **kwargs)
        return CausalLMOutputWithCrossAttentions(logits=out)


def run_onnxruntime(args, model_file, inputs, model_name, config, device):
    local_rank = get_rank()
    model_file = f'{args.save_dir}/{model_file}'
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank, 'tunable_op_enabled': args.tune})])

    ## bind inputs by using OrtValue
    #io_binding = sess.io_binding()
    #input_names = sess.get_inputs()
    #for k in input_names:
    #    np_data = inputs[k.name].cpu().numpy()
    #    x = OrtValue.ortvalue_from_numpy(np_data, 'cuda', local_rank)
    #    io_binding.bind_ortvalue_input(k.name, x)
    ## bind outputs
    outputs = sess.get_outputs()
    #for out in outputs:
    #    io_binding.bind_output(out.name, 'cuda', local_rank)

    def infer_func(input_ids, attention_mask, **kwargs):
        #sess.run_with_iobinding(io_binding)
        #output = io_binding.copy_outputs_to_cpu()
        #print('input ids: ', input_ids.shape, ' atten shape: ', attention_mask.shape)
        inputs = {'input_ids': input_ids, 'attention_mask':attention_mask}
        inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        out = sess.run(None, inputs)
        return torch.from_numpy(out[0])

    model = GenerateWrapper(infer_func, config=config)

    # warmup
    out = model.generate(**inputs, max_length=args.gen_len)

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        #y = sess.run(None, inputs)
        #sess.run_with_iobinding(io_binding)
        output = model.generate(**inputs, max_length=args.gen_len)

        if local_rank == 0 and i % interval == 0:
            cost_time = time.time() - end
            print(f'iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()

    return output

def get_dummy_inputs(model_name, batch, seq_len, past_seq_len, config, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch, seq_len), dtype=torch.int64, device=device)
    input_str = 'hello world' * (seq_len // 2)
    input_ids = tokenizer(input_str, return_tensors='pt')
    att_mask = input_ids['attention_mask'].to(device)
    input_ids = input_ids['input_ids'].to(device)

    inputs = (input_ids, att_mask)
    input_names = ['input_ids', 'attention_mask']
    return {k: v for k, v in zip(input_names, inputs)}, input_names
   
def get_bloom_model(args, name):
    config = AutoConfig.from_pretrained(name)
    #config.n_layer=2
    config.use_cache=False
    model = BloomForCausalLM.from_pretrained(name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return config, model, tokenizer

def run_torch_model(args, model, config, inputs, device):
    if args.fp16:
        model.half()

    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def infer_torch(input_ids, attention_mask, **kwargs):
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        #inputs.update(kwargs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        return out.logits

    gen_model = GenerateWrapper(infer_torch, config=config)

    # try warmup
    with torch.no_grad():
        output = gen_model.generate(**inputs, max_length=args.gen_len)

    print('output: ', output.shape, ' dtype:', output.dtype, ' dev:', output.device)

    local_rank = get_rank()

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        with torch.autograd.profiler.emit_nvtx(args.profile):
            with torch.no_grad():
                output = gen_model.generate(**inputs, max_length=args.gen_len)

        if local_rank == 0 and i % interval == 0:
            cost_time = time.time() - end
            print(f'[torch] iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()
    return output


def export_model(args, model, config, world_size, local_rank, inputs, input_names, output_names, model_out_file, tmp_out_file, opt_out_file): 
    # sync all process
    print('start to export model')

    dyn_axes = {'input_ids': {0: 'batch', 1: 'seq_len'}, 'attention_mask': {0: 'batch', 1: 'seq_len'}}
    print(f'rank: {local_rank} start to export onnx')
    torch.onnx.export(
            model,
            f=tmp_out_file,
            args=inputs,
            input_names=input_names,
            output_names=output_names,
            opset_version=15,
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            custom_opsets={'com.microsoft':1},
            export_params=True, # need these config to keep result same as torch
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            dynamic_axes=dyn_axes,
        )

    onnx_model = onnx.load(tmp_out_file)
    onnx.save(onnx_model, model_out_file, save_as_external_data=True, location=f'{model_out_file}.data')
    print(f'rank: {local_rank} export to onnx done.')

    ## use fusion to optimize model
    model_type = 'bert'
    opt_option=FusionOptions(model_type)
    #opt_option.enable_attention=False
    #opt_option.enable_bias_skip_layer_norm=False
    optimizer = optimize_by_fusion(
            onnx.load(model_out_file), 
            model_type=model_type,
            num_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            optimization_options=opt_option
        )
    if args.fp16:
        optimizer.convert_float_to_float16(use_symbolic_shape_infer=True, keep_io_types=True)

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
    device=init_dist(args)
    batch=args.batch
    seq_len=args.seq_len
    model_name = f'bigscience/{args.model}'

    local_rank = get_rank()
    world_size = get_size()

    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)

    config, model, tokenizer = get_bloom_model(args, model_name)

    if model is not None:
        model.eval()
        model.requires_grad_(False)

    #prompt = ['Morning, I want to', 'My dog is', 'If you want to, I']
    prompt = ['My dog is']
    #prompt = ['Morning, I want to']
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k : v for k,v in inputs.items()}
    input_names = list(inputs.keys())
    output_names = ['output']

    # try to export model to onnx
    model_out_file = f'rank-{local_rank}-{args.output}'
    tmp_file = f'tmp-{model_out_file}'
    opt_out_file = f'opt-{model_out_file}'

    if args.export:
        export_model(args, model, config, world_size, local_rank, inputs, input_names, output_names, model_out_file, tmp_file, opt_out_file)

    if not args.no_torch_infer:
        torch_out = run_torch_model(args, model, config, inputs, device)
        gen_torch_out = tokenizer.batch_decode(torch_out, skip_special_token=True)
        print('torch res: ', gen_torch_out)

    if not args.no_ort_infer:
        ort_out = run_onnxruntime(args, opt_out_file, inputs, model_name, config, device)
        gen_ort_out = tokenizer.batch_decode(ort_out, skip_special_token=True)
        print('ort res: ', gen_ort_out)

    if not args.no_torch_infer and not args.no_ort_infer:
        compare(args, torch_out.cpu().numpy(), ort_out.cpu().numpy())


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--gen-len', type=int, default=256)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--no-torch-infer', action='store_true', default=False)
    parser.add_argument('--no-ort-infer', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
