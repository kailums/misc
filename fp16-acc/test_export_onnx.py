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
    so.log_severity_level = 4 # 0 for verbose

    return so

def run_onnxruntime(args, model_file, inputs, local_rank):
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('CUDAExecutionProvider',{'device_id':local_rank})])
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

class Attention(torch.nn.Module):
    def __init__(self, in_features, num_heads, kdim=256):
        super().__init__()
        self.heads = num_heads
        self.wq = torch.nn.Linear(in_features, in_features)
        self.wk = torch.nn.Linear(in_features, in_features)
        self.wv = torch.nn.Linear(in_features, in_features)
        self.wo = torch.nn.Linear(in_features, in_features)
        #self.atten = torch.nn.MultiheadAttention(in_features, num_heads, batch_first=True)

    def split(self, t):
        b, seq, dim = t.size()
        d_head = dim // self.heads
        t = t.view(b, seq, self.heads, d_head).transpose(1, 2)
        return t

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # do attention
        b, heads, seq, dim = q.size()
        k_t = k.transpose(2,3) # trans k to (b, h, dim, s)

        score = (q @ k_t) / math.sqrt(dim)  # shape is (b, h, s, s)
        score = F.softmax(score, dim=-1, dtype=torch.float32).to(score.dtype)

        out = score @ v  # out shape is (b, h, s, dim)

        out = out.transpose(1,2).contiguous().view(b, seq, dim  * heads)

        out = self.wo(out)
        
        return out

class MLP(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.h_to_4h = torch.nn.Linear(in_features, in_features * 4)
        self.h4_to_h = torch.nn.Linear(in_features * 4, in_features)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.h_to_4h(x)
        x = self.act(x)
        x = self.h4_to_h(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, in_features, num_heads, use_residual, use_layernorm):
        super().__init__()
        self.use_residual = use_residual
        self.use_ln = use_layernorm
        self.ln = torch.nn.LayerNorm(in_features)
        self.att = Attention(in_features, num_heads)
        self.ln2 = torch.nn.LayerNorm(in_features)
        self.mlp = MLP(in_features)

    def forward(self, x):
        residual = x

        if self.use_ln:
            x = self.ln(x)

        x = self.att(x)

        if self.use_residual:
            x = x + residual

        residual = x
        if self.use_ln:
            x = self.ln2(x)

        x = self.mlp(x)

        if self.use_residual:
            x = x + residual

        return x


class TestModule(torch.nn.Module):
    def __init__(self, num_layers, in_features, num_heads, use_residual=False, use_layernorm=False):
        super().__init__()
        m = []
        for _ in range(num_layers):
            m.append(Block(in_features, num_heads, use_residual, use_layernorm))
        self.m = torch.nn.ModuleList(m)

    def forward(self, x):
        for l in self.m:
            x = l(x)
       
        return x


def main():
    batch = 1
    seqlen = 128
    features = 1024
    num_layers = 16
    num_heads = 16
    use_residual = True
    use_layernorm = False
    torch.cuda.manual_seed(42)
    device = torch.device('cuda:0')

    model = TestModule(num_layers, features, num_heads, use_residual, use_layernorm)
    model.requires_grad_(False)
    model.half()
    model.to(device)

    x = torch.randn(1, seqlen, features, device=device, dtype=torch.float16)

    inputs = (x,)
    input_names = ['input']
    output_names = ['out']

    out1 = model(*inputs)

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

    out2 = run_onnxruntime(None, tmp_file, {k: v for k, v in zip(input_names, inputs)}, 0)

    print(f'out shape: {out1.shape}, out type: {out1.dtype}')

    o1 = out1.cpu().numpy()
    
    if np.allclose(o1, out2[0]):
        print('ALL SAME')
    else:
        diff = abs(o1 - out2[0])
        print(f'not SAME, max diff: {diff.max()}')


if __name__ == '__main__':
    main()
