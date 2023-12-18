# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import sys

import torch

sys.path.append('../..')
from common.lookahead_cache import LookaheadCache

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import AutoTokenizer

from .benchmark import Benchmark


class LlameBenchmark(Benchmark):

    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from models.llama.modeling_llama_flash_triton import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_dir
                                                 , cache_dir='../'
                                                 , torch_dtype=torch.float16
                                                 , low_cpu_mem_usage=True
                                                 , device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
        model.lookahead_cache = lookahead_cache
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = None


model_dir = '/mntnlp/common_base_model/llama2-7b-chat'
prompt_dir = '/mntnlp/nanxiao/lookahead_benchmark/dolly_15k_llama2_13b_chat.jsonl'

worker = LlameBenchmark(log_dir='llama_benchmakr')
worker.initialize(model_dir=model_dir, token_dir=model_dir)
worker.load_prompts(prompt_dir=prompt_dir)

# runable check
prompt = '杭州在哪里？'
decoding_mode = 'hier'
max_length = 256
worker.chat(prompt,
            max_length=max_length,
            use_lookahead=False,
            decoding_length=15,
            branch_length=4,
            decoding_mode=decoding_mode,
            debug_lookahead=False)

# test correctness with lookahead decoding
worker.batch_chat(worker.prompts[:10],
                  max_length=max_length,
                  decoding_length=15,
                  branch_length=4,
                  decoding_mode=decoding_mode,
                  debug_lookahead=False,
                  erase=True,
                  batch_size=1)

chat_count = 100
warmup_count = 1000
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  decoding_mode=decoding_mode, sizes=[64], lens=[0], max_length=max_length)
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  decoding_mode=decoding_mode, sizes=[64], lens=[12], max_length=max_length)
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  decoding_mode=decoding_mode, sizes=[64], lens=[12], max_length=max_length)
