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
from onnxruntime.training.ortmodule import ORTModule, DebugOptions, LogLevel

from onnxruntime.training.ortmodule._utils import _ortvalues_to_torch_tensor

from transformers import AutoConfig, AutoTokenizer, AutoModel, GenerationMixin, PreTrainedModel, GenerationConfig
from transformers import BertModel, BloomForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from models.modeling_bloom import BloomForCausalLM as BloomForCausalLMDist


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
        self.main_input_name = 'input_ids'
        self.infer_func = infer_func
        self.one = nn.Parameter(torch.tensor([0]), requires_grad=False)

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, input_ids, attention_mask, **kwargs):
        ret = {'input_ids': input_ids, 'attention_mask': attention_mask}
        ret.update(kwargs)
        return ret

    def forward(self, input_ids, attention_mask, **kwargs):
        inputs = {'input_ids': input_ids, 'attention_mask':attention_mask}
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            inputs['past_key_values'] = kwargs['past_key_values']
            input_seq_len = input_ids.shape[-1]
            if input_seq_len != 1:
                input_seq1 = input_ids[:,-1][:,None]
                inputs['input_ids'] = input_seq1

        if 'use_cache' in kwargs:
            inputs['use_cache'] = kwargs['use_cache']

        logits, past_kv = self.infer_func(**inputs)
        return CausalLMOutputWithCrossAttentions(logits=logits, past_key_values=past_kv)

numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


def run_onnxruntime(args, model_file, cache_model_file, inputs, model_name, config, device):
    local_rank = get_rank()
    model_file = f'{args.save_dir}/{model_file}'
    cache_model_file = f'{args.save_dir}/{cache_model_file}'
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank, 'tunable_op_enabled': args.tune})])
    if args.cache:
        cache_sess = ort.InferenceSession(cache_model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank, 'tunable_op_enabled': args.tune})])

    inputs = {k: v.to(device) for k, v in inputs.items()}

    def infer_func(**inputs):
        torch.cuda.nvtx.range_push('onnx-forward')
        # bind inputs by using buffer from torch tensor
        use_cache = inputs['use_cache'] if 'use_cache' in inputs else False
        working_sess = cache_sess if 'past_key_values' in inputs and use_cache else sess
        if 'past_key_values' in inputs:
            past_key_values = {}
            for i, kv in enumerate(inputs['past_key_values']):
                past_key_values[f'past_{i}_key'] = kv[0]
                past_key_values[f'past_{i}_value'] = kv[1]

        io_binding = working_sess.io_binding()
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        # rebind inputs and output
        input_names = working_sess.get_inputs()
        for k in input_names:
            torch_tensor = inputs[k.name] if k.name in inputs else past_key_values[k.name]
            io_binding.bind_input(
                    name=k.name, device_type='cuda', device_id=local_rank,
                    element_type=torch_to_numpy_dtype_dict[torch_tensor.dtype],
                    shape=tuple(torch_tensor.shape),
                    buffer_ptr=torch_tensor.data_ptr()
                    )
            #np_data = torch_tensor.cpu().numpy()
            #x = OrtValue.ortvalue_from_numpy(np_data, 'cuda', local_rank)
            #io_binding.bind_ortvalue_input(k.name, x)
        # bind outputs
        outputs = working_sess.get_outputs()
        for out in outputs:
            io_binding.bind_output(out.name, 'cuda', local_rank)

        working_sess.run_with_iobinding(io_binding)
        outputs = io_binding.get_outputs_as_ortvaluevector()
        out = _ortvalues_to_torch_tensor(outputs) 
        #out = io_binding.copy_outputs_to_cpu()
        #logits = torch.empty_like(out[0]).copy_(out[0])
        logits = out[0]
        #logits = torch.from_numpy(out[0]).to(device)
        past_key_values = None
        if len(out) > 1:
            present = out[1:]
            past_key_values = []
            for i in range(config.n_layer):
                key = present[i*2]
                value = present[i*2+1]

                past_key_values.append([key, value])

        torch.cuda.nvtx.range_pop()
        return logits, past_key_values

    model = GenerateWrapper(infer_func, config=config)
    model.to(device)

    # warmup
    torch.cuda.nvtx.range_push('onnx-generate')
    output = model.generate(**inputs, max_length=args.gen_len, use_cache=args.cache)
    torch.cuda.nvtx.range_pop()

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        torch.cuda.nvtx.range_push('onnx-generate')
        output = model.generate(**inputs, max_length=args.gen_len, use_cache=args.cache)
        torch.cuda.nvtx.range_pop()

        if i % interval == 0:
            cost_time = time.time() - end
            print(f'iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()

    return output

def run_torch_model(args, model, config, inputs, device):
    if args.fp16:
        model.half()

    print('run torch with half, on dev: ', device)
    model.to(device)
    if args.compile:
        model = torch.compile(model)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    def infer_torch(**inputs):
        torch.cuda.nvtx.range_push('torch-forward')
        out = model(**inputs)
        torch.cuda.nvtx.range_pop()
        return out.logits, out.past_key_values

    gen_model = GenerateWrapper(infer_torch, config=config)
    gen_model.to(device)

    # try warmup
    with torch.no_grad():
        torch.cuda.nvtx.range_push('torch-generate')
        output = gen_model.generate(**inputs, max_length=args.gen_len, use_cache=args.cache)
        torch.cuda.nvtx.range_pop()

    print('output: ', output.shape, ' dtype:', output.dtype, ' dev:', output.device)

    local_rank = get_rank()

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        with torch.no_grad():
            torch.cuda.nvtx.range_push('torch-generate')
            output = gen_model.generate(**inputs, max_length=args.gen_len, use_cache=args.cache)
            torch.cuda.nvtx.range_pop()

        if i % interval == 0:
            cost_time = time.time() - end
            print(f'[torch] iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()
    return output

def run_ort_model(args, model, config, inputs, device):
    if args.fp16:
        model.half()

    model.to(device)
    model = ORTModule(model, DebugOptions(save_onnx=True, log_level=LogLevel.WARNING, onnx_prefix="ort-bloom"))
    model.eval()

    inputs = {k: v.to(device) for k, v in inputs.items()}

    def infer_torch(**inputs):
        out = model(**inputs)
        return out.logits, out.past_key_values

    gen_model = GenerateWrapper(infer_torch, config=config)
    gen_model.to(device)

    # try warmup
    with torch.no_grad():
        output = gen_model.generate(**inputs, max_length=args.gen_len, use_cache=args.cache)

    print('output: ', output.shape, ' dtype:', output.dtype, ' dev:', output.device)

    local_rank = get_rank()

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        with torch.no_grad():
            output = gen_model.generate(**inputs, max_length=args.gen_len, use_cache=args.cache)

        if i % interval == 0:
            cost_time = time.time() - end
            print(f'[ort-model] iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()
    return output

def get_dummy_inputs(args, tokenizer, batch, seq_len, past_seq_len, config, device):
    start_prompt = ['hello'] * seq_len
    prompt = [' ' .join(start_prompt)] * batch
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_names = ['input_ids']
    output_names = ['output']
    if args.cache:
        for i in range(config.n_layer):
            output_names.append(f'present_{i}_key')
            output_names.append(f'present_{i}_value')

    # convert to dict
    dyn_axes = {'input_ids': {0: 'batch', 1: 'seq_len'}, 'attention_mask': {0: 'batch', 1: 'seq_len'}, 'output': {0: 'batch', 1: 'seq_len'}}

    if past_seq_len > 0:
        # attention mash should be (b, seqlen+past_len)
        # here re-create attention_mask
        inputs['attention_mask'] = torch.ones((batch, seq_len + past_seq_len), device=device)
        dyn_axes['attention_mask'] = {0: 'batch', 1: 'seq_past_len'}

        # create past kv: [[k,v],[k,v],...]
        # k shape: [b*n_head, head_dim, kv_len]
        # v shape: [b*n_head, kv_len, head_dim]
        world_size = get_size()
        num_heads = config.n_head
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        num_heads = num_heads // world_size
        k_shape = [batch * num_heads, head_dim, past_seq_len]
        v_shape = [batch * num_heads, past_seq_len, head_dim]
        past_key_values = []
        kv_dtype = torch.float16 if args.fp16 else torch.float32
        for i in range(config.n_layer):
            key = torch.randn(k_shape).to(kv_dtype).to(device)
            value = torch.randn(v_shape).to(kv_dtype).to(device)
            past_key_values.append([key, value])
            input_names.append(f'past_{i}_key')
            dyn_axes[f'past_{i}_key'] = {0: 'bn', 2: 'past_seq_len'}
            input_names.append(f'past_{i}_value')
            dyn_axes[f'past_{i}_value'] = {0: 'bn', 1: 'past_seq_len'}

        inputs['past_key_values'] = past_key_values

    input_names.append('attention_mask')

    return inputs, input_names, output_names, dyn_axes
   
def export_model(args, model, config, world_size, local_rank, inputs, input_names, output_names, dyn_axes, model_out_file, tmp_out_file, opt_out_file): 
    # sync all process
    barrier()
    print('start to export model')
    print('input-ids shape: ', inputs['input_ids'].shape)

    for i in range(world_size):
        if i == local_rank:
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
        barrier()

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

def get_bloom_model(args, name):
    config = AutoConfig.from_pretrained(name)
    config.n_layer=2
    config.use_cache=args.cache
    process_group=get_process_group()
    model = BloomForCausalLM.from_pretrained(name, config=config)
    #model = BloomForCausalLMDist(config=config, process_group=process_group)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return config, model, tokenizer

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
        #if args.fp16:
        #    model.to(device)
        #    model.half()

    #prompt = ['My dog is'] * batch
    start_prompt = ['hello'] * seq_len
    prompt = [' ' .join(start_prompt)] * batch
    #prompt = ['Morning, I want to']

    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k : v for k,v in inputs.items()}

    # try to export model to onnx
    model_out_file = f'rank-{local_rank}-{args.output}'
    tmp_file = f'tmp-{model_out_file}'
    opt_out_file = f'opt-{model_out_file}'
    cache_opt_out_file = f'cache_{opt_out_file}'

    if args.export:
        dummy_inputs, input_names, output_names, dyn_axes = get_dummy_inputs(args, tokenizer, 1, args.seq_len, 0, config, model.device)
        export_model(args, model, config, world_size, local_rank, dummy_inputs, input_names, output_names, dyn_axes, model_out_file, tmp_file, opt_out_file)

        if args.cache:
            # export onnx with use_cache
            dummy_inputs, input_names, output_names, dyn_axes = get_dummy_inputs(args, tokenizer, 1, 1, args.seq_len, config, model.device)
            export_model(args, model, config, world_size, local_rank, dummy_inputs, input_names, output_names, dyn_axes, model_out_file, tmp_file, cache_opt_out_file)


    if not args.no_torch_infer:
        torch_out = run_torch_model(args, model, config, inputs, device)
        gen_torch_out = tokenizer.batch_decode(torch_out, skip_special_token=True)
        if local_rank == 0:
            print('torch res: ', gen_torch_out[0])

    if not args.no_ort_infer:
        ort_out = run_ort_model(args,model, config, inputs, device)
        gen_ort_out = tokenizer.batch_decode(ort_out, skip_special_token=True)
        if local_rank == 0:
            print('OrtModule res: ', gen_ort_out[0])

    if not args.no_onnx_infer:
        ort_out = run_onnxruntime(args, opt_out_file, cache_opt_out_file, inputs, model_name, config, device)
        gen_ort_out = tokenizer.batch_decode(ort_out, skip_special_token=True)
        if local_rank == 0:
            print('onnxruntime res: ', gen_ort_out[0])


    if not args.no_torch_infer and not args.no_ort_infer:
        compare(args, torch_out.cpu().numpy(), ort_out.cpu().numpy())


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
    parser.add_argument('--gen-len', type=int, default=256)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--no-torch-infer', action='store_true', default=False)
    parser.add_argument('--no-ort-infer', action='store_true', default=False)
    parser.add_argument('--no-onnx-infer', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--cache', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
