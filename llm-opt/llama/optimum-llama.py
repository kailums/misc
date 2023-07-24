from transformers import LlamaTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import onnxruntime as ort

model_path = 'decapoda-research/llama-7b-hf'

saved_name = 'llama-7b-hf'

# set use_merged = True to combine decoder_model.onnx and decoder_with_past_model.onnx into one model
#import pdb;pdb.set_trace()
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

model = ORTModelForCausalLM.from_pretrained(model_path, export=False, use_merged=False, provider='ROCMExecutionProvider', session_options=so, provider_options={'device_id':0}, use_io_binding=True)
#model.save_pretrained(saved_name)

tokenizer = LlamaTokenizer.from_pretrained(model_path)
prompt='Q: What is the largest animal?\nA:'

inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=32)

response = tokenizer.decode(outputs[0])


