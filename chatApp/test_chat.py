import torch
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from chatcli import chat_loop, generate, generate_stream


def main():
    rank = 15
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    config = LlamaConfig.from_pretrained(model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=config.torch_dtype, config=config)
  
    model.to(torch.device(rank))
  
    chat_loop(
        model,
        tokenizer,
        generate_func=generate,
        max_new_tokens=512,
    )

if __name__ == '__main__':
    main()
