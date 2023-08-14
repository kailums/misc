import torch
import onnx
from onnxscript import proto2python


onnx_model_file = './Llama-2-Onnx-7-FT-16/ONNX/LlamaV2_7B_FT_float16.onnx'


onnx_model = onnx.load(onnx_model_file, load_external_data=False)

code = proto2python(onnx_model, use_operators=True, inline_const=True)

with open('llamav2-7b-model.py', 'w') as fp:
    fp.write(code)
