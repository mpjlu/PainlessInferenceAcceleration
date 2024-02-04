
import json
from transformers import AutoTokenizer

from pia.lookahead.models.glm.tokenization_glm import GLMChineseTokenizer
from pia.lookahead.common.lookahead_cache import LookaheadCache

from benchmark import Benchmark

model_dir = '/mntnlp/nanxiao/lookahead/antglm'
tokenizer = GLMChineseTokenizer.from_pretrained(model_dir)
worker = Benchmark(eos=tokenizer.eop_token_id)

stop_words = set(tokenizer.convert_tokens_to_ids([',', '.', ' ', '\n', '，', '。','是','的']))
lookahead_cache = LookaheadCache(eos_ids=[tokenizer.eos_token_id], stop_words=stop_words)

put_ids = []
get_ids = []
dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/train.jsonl'
for line in open(dataset_dir):
    line = json.loads(line)
    put_ids.append(line['ids'])

dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/test.jsonl'
for line in open(dataset_dir):
    line = json.loads(line)
    get_ids.append(line['ids'])

for max_node_rate in [1,2,4,8,16,32,64,128]:
    worker.perf_check_trie(lookahead_cache, put_ids, get_ids[:500], max_node_rate=max_node_rate, decoding_length=128, branch_length=32, edl=8)