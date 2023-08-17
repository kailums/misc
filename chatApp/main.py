import os
import argparse
import torch
from inference import chat_loop
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM

def main(args):
    rank = 15
    model_id = args.model
    config = LlamaConfig.from_pretrained(model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=config.torch_dtype, config=config)
  
    model.to(torch.device(rank))
  
    chat_loop(
        model,
        tokenizer,
        max_new_tokens=128
    )


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
