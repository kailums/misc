import onnx
import onnxruntime as ort
from onnxruntime import OrtValue

# We use ONNX opset 15 to define the function below.
import onnxscript
from onnxscript import FLOAT, FLOAT16
from onnxscript import opset15 as op
from onnxscript import script
from onnxscript.values import Opset

import torch
import numpy as np
import onnx.numpy_helper as numpy_helper
import argparse
import psutil
import time
from mpi4py import MPI


op_domain_ms = Opset("com.microsoft", 1)
op_domain_custom = Opset("com.custom", 1)

# TODO: need to create this onnx domain for LayerNorm, otherwise it will raise error for can't find LayerNorm in ort.
op_domain_onnx = Opset("", 1)

# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script(op_domain_ms)
def AllReduce(X):
    """use AllReduce from ms domain."""
    return op.Identity(X)

@script(op_domain_onnx)
def LayerNormalization(x, scale, bias=None, axis: int=-1, epsilon: float=1e-5, stash_type: int=1):
    """ 
    layernorm is in onnx opset17, this function will call into onnxruntime.
    bias can be optional. use op.OptionalHasElement to check if it is None
    """
    return op.Identity(x)

@script(op_domain_ms, default_opset=op)
def SkipLayerNormalization(x, skip, gamma, beta=None, bias=None, epsilon: float=1e-5):
    '''
    this function is not use 'op' set op, so need to set 'default_opset'
    if change x = x+skip to x = op.Add(x, skip), then not need to set default opset.
    '''
    x = x + skip
    # TODO: optional input is not well supported
    #if bias != None:
    #    x = x + bias
    out = LayerNormalization(x, gamma, beta, epsilon)
    # TODO: optional output is not well supported.
    #       here just return gamma and beta to much signature of SkipLayerNorm
    return out, gamma, beta, x

@script(op_domain_ms)
def DummyGemm(x: FLOAT[None], w: FLOAT[None], alpha: float, beta: float, transA: int, transB: int) -> FLOAT[None]:
    x = op.Gemm(x, w, alpha=alpha, beta=beta, transA=transA, transB=transB)
    #x = op.MatMul(x, w)
    return x

@script(op_domain_custom)
def DummyMatMul(x, w):
    x = op.MatMul(x, w)
    return x

@script(op_domain_custom)
def gelu(x):
    x = x * 0.5 * (1.0 + op.Tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    return x

@script(op_domain_custom)
def fuse_matmul_gelu_matmul(x, w1, b1, w2, b2):
    x = op.MatMul(x, w1) + b1
    x = x * 0.5 * (1.0 + op.Tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    return op.MatMul(x, w2) + b2


def create_model(batch, seqlen, hidden, out_file_name, dtype=np.float32):
    num_layers = 2
    # We use the script decorator to indicate that
    # this is meant to be translated to ONNX.
    weight_h_h = np.random.rand(hidden, hidden).astype(dtype)
    bias_h_h = np.random.rand(hidden).astype(dtype)

    weight_h_4h = np.random.rand(hidden, hidden * 4).astype(dtype)
    bias_h_4h = np.random.rand(hidden * 4).astype(dtype)

    weight_4h_h = np.random.rand(hidden * 4, hidden).astype(dtype)
    bias_4h_h = np.random.rand(hidden).astype(dtype)

    np_ln_scale = np.random.rand(hidden).astype(dtype)
    np_ln_bias = np.random.rand(hidden).astype(dtype)

    input_types = [FLOAT[batch, seqlen, hidden]]
    output_types = [FLOAT[batch, seqlen, hidden]]
    if dtype == np.float16:
      input_types = [FLOAT16[batch, seqlen, hidden]]
      output_types = [FLOAT16[batch, seqlen, hidden]]

    @script(op_domain_onnx, default_opset=op)
    def block(
            X,
    ):
        w_h2h = op.Constant(value=numpy_helper.from_array(weight_h_h))
        b_h2h = op.Constant(value=numpy_helper.from_array(bias_h_h))
        w_h4h = op.Constant(value=numpy_helper.from_array(weight_h_4h))
        b_h4h = op.Constant(value=numpy_helper.from_array(bias_h_4h))
        w_4hh = op.Constant(value=numpy_helper.from_array(weight_4h_h))
        b_4hh = op.Constant(value=numpy_helper.from_array(bias_4h_h))

        ln_scale = op.Constant(value=numpy_helper.from_array(np_ln_scale))
        ln_bias = op.Constant(value=numpy_helper.from_array(np_ln_bias))

        # construct model
        matmul = op.MatMul(X, w_h2h) + b_h2h
        ar = AllReduce(matmul)
        ln = LayerNormalization(ar, ln_scale, ln_bias)
        #out = fuse_matmul_gelu_matmul(ln, w_h4h, b_h4h, w_4hh, b_4hh)
        out = op.Reshape(ln, op.Constant(value_ints=[batch*seqlen, hidden]))
        out = DummyGemm(out, w_h4h, 1., 0., 0, 0) + b_h4h
        #out = op.MatMul(out, w_h4h) + b_h4h
        out = gelu(out)
        out = DummyGemm(out, w_4hh, 1.0, 0.0, 0, 0) + b_4hh
        out = op.Reshape(out, op.Constant(value_ints=[batch, seqlen, hidden]))
        out = AllReduce(out)
        out,_,_,out2 = SkipLayerNormalization(out, ln, ln_scale, ln_bias, None)
        
        out = op.MatMul(out, w_h2h) + out2
        return out

    @script(op_domain_onnx, default_opset=op)
    def sample_model(x):
        for i in range(num_layers):
            x = block(x)

        return x


    # onnx_model is an in-memory ModelProto
    #onnx_model = sample_model.to_model_proto(
    onnx_model = block.to_model_proto(
                                  input_types=input_types,
                                  output_types=output_types
                              )

    # Save the ONNX model at a given path
    onnx.save(onnx_model, out_file_name, save_as_external_data=True, location=f'{out_file_name}.data')


def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    if local_rank == 0 and args.log:
        so.log_severity_level = 0
        ort.set_default_logger_severity(0)  # open log

    if args.ort_opt:
        so.optimized_model_filepath = f'ort-opted-rank-{local_rank}-{args.onnx_file_name}'

    if args.profile:
        so.enable_profiling = args.profile
        so.profile_file_prefix=f'ort-profile-rank{local_rank}'

    custom_op_lib_path = "librocm_custom_op_library.so"
    #so.register_custom_ops_library(custom_op_lib_path)

    return so

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

    device = torch.device(local_rank)
    return device

def get_rank():
    comm = MPI.COMM_WORLD
    return comm.Get_rank()

def get_size():
    comm = MPI.COMM_WORLD
    return comm.Get_size()


def run_onnxruntime(args, model_file, inputs):
    local_rank = get_rank()
    model_file = f'{args.save_dir}/{model_file}'
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank, 'tunable_op_enable': args.tune, 'tunable_op_tuning_enable': args.tune})], disabled_optimizers=['BiasSoftmaxFusion'])
    io_binding = sess.io_binding()

    # bind inputs by using OrtValue
    input_names = sess.get_inputs()
    for k in input_names:
        np_data = inputs[k.name]
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


def main(args):
    batch = 16
    seqlen = 128
    hidden = 768
    onnx_file_name = args.onnx_file_name
    dtype=np.float32
    if args.fp16:
        dtype=np.float16

    if args.export:
        create_model(batch, seqlen, hidden, onnx_file_name, dtype)

    X = np.random.rand(batch, seqlen, hidden).astype(dtype)
    inputs = {'X': X}

    out = run_onnxruntime(args, onnx_file_name, inputs)

def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--onnx-file-name', type=str, default='bert_mlp.onnx')
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
