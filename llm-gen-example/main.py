import torch
import os
import random
import argparse
from transformers import AutoConfig, AutoTokenizer
from transformers import MistralForCausalLM
from huggingface_hub import snapshot_download
import pickle
import onnxruntime as ort
from time import perf_counter

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

#ort.set_default_logger_severity(0)
#ort.set_default_logger_verbosity(1000)


def vllm_infer(model_id, prompt, gen_tokens):
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    print('lora path: ', lora_path)

    llm = LLM(
        model=model_id,
        tokenizer=model_id,
        tensor_parallel_size=1,
        seed=42,
        trust_remote_code=True,
        dtype="float16",
        backend=os.environ.get("BACKEND", "torch"),
        enforce_eager=True,
        enable_lora=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        use_beam_search=False,
        ignore_eos=False,
        max_tokens=gen_tokens,
        top_p=1.0,
        top_k=-1,
    )

    st = perf_counter()
    output = llm.generate(prompt, sampling_params, lora_request=LoRARequest("sql-lora", 1, lora_path))
    cost = perf_counter() - st
    print('generate cost: ', cost)

    out_id = 0
    print('[prompt] ', output[out_id].prompt)
    print('[output]: ', output[out_id].outputs[0].text)


def main(model_id, prompt, device_id=0, gen_tokens=128):
    device = torch.device(device_id)
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print('config: ', config)
    
    #model = MistralForCausalLM.from_pretrained(model_id, config=config, torch_dtype=config.torch_dtype)
    model = MistralForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.float16)
    model = model.to(device)

    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(inputs['input_ids'], attention_mask = inputs['attention_mask'], max_new_tokens=gen_tokens)
    gen_torch_out = tokenizer.batch_decode(output, skip_special_token=True)

    print('generate result: ', gen_torch_out[0])


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arges()
    model_id = args.model
    batch=2
    gen_tokens = 16
    device_id = 0
    torch.cuda.set_device(device_id)

    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = 'The capial of France is '
    #prompt = [prompt] * batch

    vllm_infer(model_id, prompt, gen_tokens)
