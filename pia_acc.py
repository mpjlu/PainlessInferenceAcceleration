import torch
from transformers import AutoTokenizer
import time
import json
import jsonlines
from pia.lookahead.models.llama.modeling_llama import LlamaForCausalLM
from pia.lookahead.models.baichuan2_13b.modeling_baichuan import BaichuanForCausalLM

use_lookahead = True 
#model_dir = "/data/home/pengmeng/llama-7b"
model_dir = "/root/sn-13b-v7/"
#model = BaichuanForCausalLM.from_pretrained(model_dir
model = BaichuanForCausalLM.from_pretrained(model_dir
                                         , cache_dir='./'
                                         , torch_dtype=torch.float16
                                         , low_cpu_mem_usage=True
                                         , device_map='auto'
                                         )
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

test_path_open = "./test.json"
system_prompt = "你是一个人工智能机器人，名字叫神农。\n- 神农是腾讯云智能开发的大语言模型。它的设计宗旨是有益、诚实且无害。\n- 神农可以使用用户选择的语言（英语和中文）流利地理解和交流。\n- 如果用户更正神农生成的错误答案，它会向用户致歉并接受用户的建议。"

warmup = 0
'''
for i in range(warmup):
        inputs = tokenizer(sys_prompt, return_tensors="pt")

        output_ids = model.generate(input_ids=inputs.input_ids.cuda(),
                                    attention_mask=inputs.attention_mask.cuda(),
                                    top_p = 0.8,
                                    top_k = 20,
                                    do_sample = True,
                                    max_new_tokens=256,
                                    decoding_kwargs=decoding_kwargs
                                    )
        response = tokenizer.decode(output_ids[0].tolist())[len(sys_prompt):]
        print(f'{response=}')
'''
start = time.time()
total_len = 0
inputs = []
#with open('./bad_case.jsonl', 'r') as f:
with open('./v9_8k_testdata_right.jsonl', 'r') as f:
    for items in jsonlines.Reader(f):
         for item in items:
           text = system_prompt + 'user:' + item['users'] + '\nbot:'
           # print(text)
           inputs.append(text)
    pred = []
    for line in inputs:
        text = [line]

        inputs = tokenizer(text, return_tensors="pt")

        torch.manual_seed(0)
        output_ids = model.generate(input_ids=inputs.input_ids.cuda(),
                                    attention_mask=inputs.attention_mask.cuda(),
                                    max_new_tokens=256,
                                    top_p = 0.8,
                                    top_k = 10,
                                    temperature = 1.0,
                                    do_sample = True,
                                    decoding_kwargs={'use_lookahead': use_lookahead,
                                        'decoding_length': 8,           
                                        'decoding_mode': "one",           
                                        'branch_length': 8,           
                                        }
                                    )
        response = tokenizer.decode(output_ids[0].tolist())[len(line):]
        pred += [response.replace("<unk>", "")]
        total_len += len(response)
        print(f'{response=}')
    json = []
    with jsonlines.open('./v9_8k_pia_hf_fp16.jsonl', mode='w') as writer:
        for o in pred:
            json.append({'bot': o})
        writer.write(json)
dur = time.time() - start
print("total_time:", dur, "  total_generation_len:", total_len)
