import torch
from torch.nn import functional as F
import onnx
import onnxruntime as ort
from onnxruntime import OrtValue
import psutil
import numpy as np
import math
from transformers import BloomConfig, BloomModel, AutoTokenizer
from models.modeling_bloom import BloomModel as BloomModelLocal


def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose

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


class TestModule(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.ln_f = torch.nn.LayerNorm(features)

    def forward(self, x):
        return self.ln_f(x)
       


def main():
    batch = 8
    seqlen = 128
    features = 1024
    torch.cuda.manual_seed(42)
    device = torch.device('cuda:0')
    r = torch.rand((1,))
    r = r * 256
    print('r is: ', r)

    state_dict = torch.load('last_ln.pt')

    model = TestModule(features)
    model.requires_grad_(False)

    model.to(device)

    #model.ln_f.load_state_dict(state_dict['ln_state_dict'])
    for p in model.parameters():
        p[:] = r[0]

    #data = state_dict['hidden_states']
    data = torch.rand((batch, seqlen, features)).to(device)
    data[:] = r[0]

    #model.half()
    #data = data.to(torch.float16)

    inputs = (data,)
    input_names = ['x']
    inputs = {k:v for k, v in zip(input_names, inputs)}
    output_names = ['out']

    out1 = model(data)

    tmp_file = 'test.onnx'

    torch.onnx.export(
            model,
            f=tmp_file,
            args=inputs,
            input_names=input_names,
            output_names=output_names,
            opset_version=15,
            #verbose=True,
            #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            #custom_opsets={'com.microsoft':1},
            export_params=True, # need these config to keep result same as torch
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
        )

    out2 = run_onnxruntime(None, tmp_file, inputs, 0)

    torch.save({'torch_out':out1, 'ort_out': out2[0]}, 'output-diff.pt')

    print(f'out shape: {out1.shape}, out type: {out1.dtype}')

    o1 = out1.cpu().numpy()
    
    if np.allclose(o1, out2[0]):
        print('ALL SAME')
    else:
        diff = abs(o1 - out2[0])
        print(f'not SAME, max diff: {diff.max()}')


if __name__ == '__main__':
    main()
