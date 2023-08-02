import torch
from typing import List
import os
import json
import psutil
import time
import argparse
from sentencepiece import SentencePieceProcessor
import numpy as np

import onnx
import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions

ORT_TYPE_TO_NP_TYPE = {
        'tensor(float16)': np.float16,
        'tensor(float32)': np.float32,
        'tensor(int8)': np.int8,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64,
        'tensor(bool)': bool,
        }

ORT_TYPE_TO_TORCH_TYPE = {
        'tensor(float16)': torch.float16,
        'tensor(float32)': torch.float32,
        'tensor(int8)': torch.int8,
        'tensor(int32)': torch.int32,
        'tensor(int64)': torch.int64,
        'tensor(bool)': torch.bool,
        }



class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        t = torch.tensor(t).tolist()
        return self.sp_model.decode(t)

def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    if args.logging:
        so.log_severity_level = 0
        ort.set_default_logger_severity(0)  # open log

    if args.ort_opt:
        so.optimized_model_filepath = f'ort-opted-rank-{local_rank}-{args.output}'

    if args.profile:
        so.enable_profiling = args.profile
        so.profile_file_prefix=f'ort-profile-rank{local_rank}'

    provider_opt = {'device_id': local_rank, 'tunable_op_enable': args.tune, 'tunable_op_tuning_enable': args.tune}

    return so, provider_opt

def setup_tokenizer(args):
    # load tokenizer
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    return tokenizer

class OrtModel:
    def __init__(self, args, tokenizer, local_rank):
        sess_opt, provider_opt = setup_session_option(args, local_rank)
        self.sess = ort.InferenceSession(args.model_file, sess_options=sess_opt, providers=[('ROCMExecutionProvider', provider_opt)])

        for i in self.sess.get_inputs():
            if i.name == 'x':
                input_x = i
            elif i.name == 'attn_mask':
                input_attn_mask = i
            elif i.name == 'k_cache':
                k_cache = i

        self.tokenizer = tokenizer
        hidden_size = input_x.shape[2]
        self.max_seq_len = input_attn_mask.shape[1]
        self.n_layers = k_cache.shape[1]
        self.n_heads = k_cache.shape[3]
        self.dtype = torch.float16 if input_x.type == 'tensor(float16)' else torch.float32
        self.attn_mask_shape = input_attn_mask.shape
        self.head_dim = hidden_size // self.n_heads

        # load embedding
        self.emb = torch.nn.Embedding(tokenizer.n_words, hidden_size)
        self.emb.load_state_dict(torch.load(args.emb_file))
        self.emb.eval()
        self.emb.requires_grad_(False)

        self.device = torch.device(local_rank)
        #self.emb.to(self.device)
        self.rank = local_rank

    def forward_with_io_binding(self, tokens, attn_mask, past_k, past_v, pos):
        x = self.emb(tokens)

        io_binding = self.sess.io_binding()
        name_map = {'x': x, 'attn_mask': attn_mask, 'k_cache': past_k, 'v_cache': past_v, 'pos': pos}
        for i in self.sess.get_inputs():
            t = name_map[i.name]
            io_binding.bind_input(
                    i.name,
                    'cuda',
                    self.rank,
                    ORT_TYPE_TO_NP_TYPE[i.type],
                    tuple(t.shape),
                    t.data_ptr()
                )

        outputs = []
        for out in self.sess.get_outputs():
            if out.name == 'logits':
                out_shape = out.shape
                out_dtype = ORT_TYPE_TO_TORCH_TYPE[out.type]
            elif out.name == 'k_out' or out.name == 'v_out':
                out_shape = [out.shape[0], out.shape[1], x.shape[1], out.shape[3], out.shape[4]]
                out_dtype = ORT_TYPE_TO_TORCH_TYPE[out.type]
            out_data = torch.empty(np.prod(out_shape), dtype=out_dtype, device=self.device)
            outputs.append(out_data.view(out_shape))
            io_binding.bind_output(
                    out.name,
                    'cuda',
                    self.rank,
                    ORT_TYPE_TO_NP_TYPE[out.type],
                    out_shape,
                    out_data.data_ptr()
                )

        io_binding.synchronize_inputs()
        self.sess.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        return outputs

    def forward(self, tokens, attn_mask, past_k, past_v, pos):
        tokens = torch.tensor(tokens)
        x = self.emb(tokens).to(self.dtype).numpy()
        x = np.expand_dims(x, axis=0)

        torch.cuda.nvtx.range_push('ort_forward')
        outputs = self.sess.run(None, {
            'x': x,
            'attn_mask': attn_mask,
            'k_cache': past_k,
            'v_cache': past_v,
            'pos': pos,
            })
        torch.cuda.nvtx.range_pop()
        return outputs

    def generate(self, prompt, max_new_tokens=32):
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        attn_mask = -10000.0 * torch.triu(torch.ones(self.attn_mask_shape), diagonal=1).to(self.dtype).numpy()

        pos = 0
        k_cache = torch.zeros((1, self.n_layers, self.max_seq_len, self.n_heads, self.head_dim), dtype=self.dtype).numpy()
        v_cache = torch.zeros((1, self.n_layers, self.max_seq_len, self.n_heads, self.head_dim), dtype=self.dtype).numpy()

        output_tokens = []
        for _ in range(max_new_tokens):
            results = self.forward(tokens, attn_mask, k_cache[:,:,:pos], v_cache[:,:,:pos], np.array(pos, dtype=np.int64))
            logits, k_out, v_out = results[:3]
            next_token = np.argmax(logits, axis=-1).astype(np.int64)
            output_tokens.extend(next_token)

            if next_token.item() == self.tokenizer.eos_id:
                break

            seq_len = k_out.shape[2]
            k_cache[:,:,pos: pos+seq_len] = k_out
            v_cache[:,:,pos: pos+seq_len] = v_out
            pos = pos + seq_len
            tokens = next_token
        return output_tokens

def run_generate(args, local_rank):
    tokenizer = setup_tokenizer(args)
    prompt='Q: What is the largest animal?\nA:'
    print('Prompt: ', prompt)
    
    ort_model = OrtModel(args, tokenizer, local_rank)
    outputs = ort_model.generate(prompt, max_new_tokens=32)

    response = tokenizer.decode(outputs)
    print('[ORT] Response:', response)


def func_benchmark(args, name, ort_model, prompt, prompt_len, gen_len):
    for _ in range(args.warm):
        torch.cuda.nvtx.range_push('generate')
        outputs = ort_model.generate(prompt, max_new_tokens=gen_len)
        torch.cuda.nvtx.range_pop()

    start = time.time()
    for _ in range(args.loop_cnt):
        torch.cuda.nvtx.range_push('generate')
        outputs = ort_model.generate(prompt, max_new_tokens=gen_len)
        torch.cuda.nvtx.range_pop()
    cost = time.time() - start
    print(f'[{name}]: prompt_len: {prompt_len}, generate_len: {gen_len}, cost: {cost / args.loop_cnt}s')
    return outputs

def run_benchmark(args, local_rank):
    tokenizer = setup_tokenizer(args)

    ort_model = OrtModel(args, tokenizer, local_rank)

    batch=1
    #prompt_len = ['32', '64', '128', '256', '512', '1024']
    prompt_len = ['2017']
    #generate_len = [1, 129]
    generate_len = [4]

    for p_len in prompt_len:
        for gen_len in generate_len:
            # generate input prompt
            with open('prompt.json') as fp:
                prompt_pool = json.load(fp);
            if p_len in prompt_pool:
                prompt = prompt_pool[p_len]
            else:
                prompt = ['Hello'] * (int(p_len) - 1)
                prompt = ' '.join(prompt)

            func_benchmark(args, "ORT", ort_model, prompt, prompt_len, gen_len)


def main(args):
    local_rank = 2

    if args.generate:
        run_generate(args, local_rank)

    if args.benchmark:
        run_benchmark(args, local_rank)


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--emb_file', type=str)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--warm', type=int, default=5)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--export', action='store_true', default=False)
    parser.add_argument('--merge', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--ort-opt', action='store_true', default=False)
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--opt-export', action='store_true', default=False)
    parser.add_argument('--ort', action='store_true', default=False)
    parser.add_argument('--torch', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
