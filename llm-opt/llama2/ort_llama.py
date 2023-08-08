import torch
from torch import nn
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

from transformers import AutoConfig, AutoTokenizer, AutoModel, GenerationMixin, PreTrainedModel, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


ORT_TYPE_TO_NP_TYPE = {
        'tensor(float16)': np.float16,
        'tensor(float)': np.float32,
        'tensor(int8)': np.int8,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64,
        'tensor(bool)': bool,
        }

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


class OrtModelForLlamaCausalLM(PreTrainedModel):
    def __init__(self, args, decoder_model, decoder_past_model, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.main_input_name = 'input_ids'
        self.one = nn.Parameter(torch.tensor([0]), requires_grad=False)
        sess_opt, provider_opt = setup_session_option(args, local_rank)
 
        self.sess = ort.InferenceSession(decoder_model, sess_options=sess_opt, providers=[('ROCMExecutionProvider', provider_opt)])
        self.sess_with_past = ort.InferenceSession(decoder_past_model, sess_options=sess_opt, providers=[('ROCMExecutionProvider', provider_opt)])

        for i in self.sess_with_past.get_inputs():
            if i.name == 'past_key_values.0.key':
                past_k = i
                break

        for i in self.sess_with_past.get_outputs():
            if i.name == 'logits':
                logits = i
                break

        config = kwargs.get('config', None)
        assert config is not None, 'need input config for OrtModelForLlamaCausalLM'
        self.n_layers = config.num_hidden_layers
        self.num_heads = past_k.shape[1]
        self.head_dim = past_k.shape[3]
        self.vocb_size = logits.shape[2]
        self.torch_dtype = config.torch_dtype
        self.rank = local_rank

    def can_generate(self):
        return True


    def forward_with_io_binding(self, input_ids, attn_mask, position_ids, past_kvs=None):
        sess = self.sess
        name_map = {'input_ids': input_ids, 'attention_mask': attn_mask, 'position_ids': position_ids}
        if past_kvs != None:
            sess = self.sess_with_past

            j = 0
            assert len(past_kvs) == self.n_layers * 2
            for i in range(self.n_layers):
                name_map[f'past_key_values.{i}.key']=past_kvs[j]
                j += 1
                name_map[f'past_key_values.{i}.value']=past_kvs[j]
                j += 1

        io_binding = sess.io_binding()
        for i in sess.get_inputs():
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
        output = torch.empty((*input_ids.shape, self.vocb_size), dtype=torch.float32, device=self.device)
        seq_len = input_ids.shape[-1]
        if past_kvs is not None:
            seq_len += past_kvs[0].shape[2]

        name_map['logits'] = output
        outputs.append(output)

        present_kv_shape = (input_ids.shape[0], self.num_heads, seq_len, self.head_dim)
        for i in range(self.n_layers):
            k = torch.empty(present_kv_shape, dtype=self.torch_dtype, device=self.device)
            name_map[f'present.{i}.key'] = k
            outputs.append(k)
            v = torch.empty(present_kv_shape, dtype=self.torch_dtype, device=self.device)
            name_map[f'present.{i}.value'] = v
            outputs.append(v)

        for out in sess.get_outputs():
            t = name_map[out.name]
            io_binding.bind_output(
                    out.name,
                    'cuda',
                    self.rank,
                    ORT_TYPE_TO_NP_TYPE[out.type],
                    tuple(t.shape),
                    t.data_ptr(),
                )

        io_binding.synchronize_inputs()
        sess.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        return outputs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:,-1:]
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:,-1].unsqueeze(-1)

        model_inputs = {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'attention_mask': attention_mask,
            }
 
        return model_inputs

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):

        results = self.forward_with_io_binding(input_ids, attention_mask, position_ids, past_key_values)
        logits, past_key_values = results[0], results[1:]

        return CausalLMOutputWithPast(
                    logits=logits,
                    past_key_values=past_key_values,
                )

