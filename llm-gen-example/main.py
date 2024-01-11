import torch
import os
import random
import argparse
from transformers import AutoConfig, AutoTokenizer
from transformers import MistralForCausalLM
import pickle

from vllm import LLM, SamplingParams


def vllm_infer(model_id, prompt, gen_tokens):
    llm = LLM(
        model=model_id,
        tokenizer=model_id,
        tensor_parallel_size=1,
        seed=42,
        trust_remote_code=True,
        dtype="float16",
        backend=os.environ.get("BACKEND", "torch"),
        #backend='ort'
        #enforce_eager=True
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

    output = llm.generate(prompt, sampling_params)

    out_id = 0
    print('[prompt] ', output[out_id].prompt)
    print('[output]: ', output[out_id].outputs[0].text)

    print('[prompt] ', output[out_id + 1].prompt)
    print('[output]: ', output[out_id + 1].outputs[0].text)


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
    batch=32
    gen_tokens = 128
    device_id = 0
    torch.cuda.set_device(device_id)

    prompt = 'Once upon a time, there was a little girl who loved to read. She loved to read so much that she would read books over and over again. She would read books that were too hard for her to understand, but she would read them anyway. She would read books that were too easy for her to understand, but she would read them anyway. She would read books that were too long for her to understand, but she would read them anyway. She would read books that were too short for her to understand, but she would read them anyway. She would read books that were too boring for her to understand, but she would read them anyway. She would read books that were too exciting for her to understand, but she would read them anyway. She would read books that were too sad for her to understand, but she would read them anyway. She would read books that were too easy for her to understand, but she would read them anyway. She would read books that were too long for her to understand, but she would read them anyway. She would read books that were too short for her to understand, but she would read them anyway. She would read books that were too boring for her to understand, but she would read them anyway. She would read books that were too exciting for her to understand, but she would read them anyway. She would read books that were too sad for her to understand, but she would read them anyway. She would read books that were too happy for her to understand, but she would read them anyway. She would read books that were too scary for her to understand, but she would read them anyway. She would read books that were too funny for her to understand, but she would read them anyway. She would read books that were too weird for her to understand, but she would read them anyway. She would read books that were too beautiful for her to understand, but she would read them anyway. She would read books that were too ugly for her to understand, but she would read them anyway. She would read books that were too smart '
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt)
    input_ids = inputs['input_ids']
    p = []
    for _ in range(batch):
        size = random.randint(8, 32)
        p.append(input_ids[:-size])

    prompt = p
    sizes = [len(p) for p in prompt]
    print('prompt sizes: ', sizes)

    prompt = tokenizer.batch_decode(prompt)

    #with open('/ws/code/vllm-ort/test-vllm/prompts.pkl', 'rb') as fp:
    #    prompt = pickle.load(fp)
    ##prompt = prompt[:2]
    #prompt = [prompt[0], prompt[0]]
    #prompt = 'Sure, I can do that. What new technology would you like me to review?,'
    prompt = ['Hello, how do you do? '] * 2
    #prompt = ' '.join(prompt)
    #main(model_id, prompt, device_id, gen_tokens)

    vllm_infer(model_id, prompt, gen_tokens)
