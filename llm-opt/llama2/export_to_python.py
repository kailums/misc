import torch
import onnx
from onnxscript import proto2python


onnx_model_file = 'LlamaV2_7B_FT_float16.onnx'


code = proto2python(onnx_model_file)

with open('llamav2-model.py', 'w') as fp:
    fp.write(code)
