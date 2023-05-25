import torch
from torch.nn import functional as F
import onnx
import onnxruntime as ort
from onnxruntime import OrtValue
import psutil
import numpy as np
import math


def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 0 # 0 for verbose
    ort.set_default_logger_severity(0)  # open log

    return so

def run_onnxruntime(args, model_file, inputs, local_rank):
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank})])
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
    return output

def main():
    batch = 4
    seqlen = 128
    features = 1024
    shape = (batch, seqlen)
    torch.cuda.manual_seed(42)
    device = torch.device('cuda:0')

    inputs = {
            'X': torch.randn((batch, seqlen), device=device, dtype=torch.float32),
            'A': torch.randn((seqlen, seqlen), device=device, dtype=torch.float32),
            'B': torch.randn((batch, seqlen), device=device, dtype=torch.float32),
            }

    tmp_file = 'function-model.onnx'

    out = run_onnxruntime(None, tmp_file, inputs, 0)

    print(f'out shape: {out[0].shape}, out type: {out[0].dtype}')
    
if __name__ == '__main__':
    main()
