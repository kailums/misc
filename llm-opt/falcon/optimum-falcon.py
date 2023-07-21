from optimum.onnxruntime import ORTModelForCausalLM

model_path='tiiuae/falcon-7b'

saved_name = 'falcon-7b-hf'

# set use_merged = True to combine decoder_model.onnx and decoder_with_past_model.onnx into one model
#import pdb;pdb.set_trace()
model = ORTModelForCausalLM.from_pretrained(model_path, export=True, use_merged=False, trust_remote_code=True)
model.save_pretrained(saved_name)

