from optimum.onnxruntime import ORTModelForCausalLM

model_path = 'decapoda-research/llama-7b-hf'

saved_name = 'llama-7b-hf'

# set use_merged = True to combine decoder_model.onnx and decoder_with_past_model.onnx into one model
#import pdb;pdb.set_trace()
model = ORTModelForCausalLM.from_pretrained(model_path, export=True, use_merged=False)
model.save_pretrained(saved_name)

