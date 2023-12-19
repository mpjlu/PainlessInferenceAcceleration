# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json
import pickle
from collections import defaultdict
from functools import reduce

import numpy as np


class Node():
    __slots__ = ['freqs', 'pairs']

    def __init__(self, freqs=None, pairs=None):
        self.freqs = freqs
        self.pairs = pairs

    def __repr__(self):
        return f'{self.freqs}->{list(self.pairs.keys())}'


class Tree():
    def __init__(self, token_id, max_node=512, max_output_node=256):
        self.token_id = token_id
        self.max_node = max_node
        self.max_output_node = max_output_node
        self.n_node = 0
        self.n_output_node = 0
        self.nodes = {}

    def put(self, token_ids, mode='output', idx=-1):
        assert mode in ('input', 'output')
        if mode == 'output':
            assert idx == -1
        else:
            assert idx >= 0
        self._put(token_ids, self.nodes, mode=mode, idx=idx, freq=1.0)
        return None

    def _put(self, token_ids, pairs, mode='output', freq=1.0, idx=-1):

        while True:
            if len(token_ids) == 0:
                break
            t = token_ids[0]
            pair = pairs.get(t, None)
            if pair is None:
                node = self._pack(token_ids, idx=idx, freq=freq)
                pairs.update(node)
                self.n_node += len(token_ids)
                if mode == 'output':
                    self.n_output_node += len(token_ids)
                break

            pair.freqs[idx] = pair.freqs.get(idx, 0.0) + freq
            pairs = pair.pairs
            token_ids = token_ids[1:]
        return None

    def _pack(self, token_ids, idx=-1, freq=1.0):
        ps = {}
        for token in token_ids[::-1]:
            freqs = {idx: freq}
            p = Node(freqs=freqs, pairs=ps)
            ps = {token: p}
        return ps

    def get(self, token_ids, max_size=64, max_length=8, min_input_size=0, min_output_size=0, output_weight=1e-4,
            mode='mix', idx=0):
        assert mode in ('input', 'output', 'mix')

        match_token_id, pairs = self._match(token_ids, mode=mode, idx=idx)
        if len(pairs) == 0:
            token_id = token_ids[-1] if len(token_ids) > 0 else self.token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0,0]

        freqs = []
        self._get_freqs(pairs, freqs, idx, output_weight)

        min_mix_freq = 1e9
        min_input_freq = 1e9
        min_output_freq = 1e9
        if mode == 'input':
            output_weight = 0.0
            size = len([x for x in freqs if x[1] > 0])
            if size > max_size:
                input_freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
                min_input_freq = input_freqs[min_input_size - 1][1]
            else:
                min_input_freq = 0.0
        elif mode == 'output':
            output_weight = 1.0
            size = len([x for x in freqs if x[2] > 0])
            if size > max_size:
                output_freqs = sorted(freqs, key=lambda x: x[2], reverse=True)
                min_output_freq = output_freqs[min_output_size - 1][2]
            else:
                min_output_freq = 0.0
        else:
            size = len([x for x in freqs if x[1] > 0 or x[2] > 0])
            if size > max_size:
                indices = set()
                if min_input_size > 0:
                    input_freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
                    min_input_freq = input_freqs[min_input_size - 1][1]
                    indices.update([x[0] for x in input_freqs[:min_input_size]])

                if min_output_size > 0:
                    output_freqs = sorted(freqs, key=lambda x: x[2], reverse=True)
                    min_output_freq = output_freqs[min_output_size - 1][2]
                    indices.update([x[0] for x in output_freqs[:min_output_size]])

                if len(indices) < max_size:
                    mix_freqs = sorted(freqs, key=lambda x: x[3], reverse=True)
                    rest_size = max_size - len(indices)
                    indices.update([x[0] for x in mix_freqs[:rest_size]])
                    cur_size = len(indices)
                    for i in range(rest_size, min(rest_size + max_size, size)):
                        if mix_freqs[i][0] in indices:
                            continue
                        cur_size += 1
                        if cur_size >= max_size:
                            x = mix_freqs[i]
                            min_mix_freq = x[3]
                            break
            else:
                min_mix_freq = 0.0

        mask = np.zeros((max_size, max_size), dtype=np.int64)
        mask[:, 0] = 1
        ids = [match_token_id or self.token_id]
        sizes = [0, 0]
        self._ravel(pairs, ids, mask, -1,
                    max_size=max_size,
                    max_length=max_length - 1,
                    min_output_freq=min_output_freq,
                    min_input_freq=min_input_freq,
                    min_mix_freq=min_mix_freq,
                    sizes=sizes,
                    output_weight=output_weight,
                    mode=mode,
                    idx=idx)
        size = len(ids)

        mask = mask[:size, :size]
        return ids, mask, sizes

    def _get_freqs(self, pairs, freqs, idx, output_weight):
        for tid, pair in pairs.items():
            fo = pair.freqs.get(-1, 0.0)
            fi = pair.freqs.get(idx, 0.0)
            if fo > 0 or fi > 0:
                fm = (1.0 - output_weight) * fi + output_weight * fo
                freqs.append([len(freqs), fi, fo, fm])
                if len(pair.pairs) > 0:
                    self._get_freqs(pair.pairs, freqs, idx, output_weight)

    def get_one_branch(self, token_ids, max_length=8, mode='output', idx=-1):
        assert mode in ('input', 'output', 'mix')

        match_token_id, pairs = self._match(token_ids, mode=mode)
        if len(pairs) == 0:
            token_id = token_ids[-1] if len(token_ids) > 0 else self.token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0,0]

        ids = [match_token_id]
        length = 0
        while True:
            if len(pairs) == 0 or length >= max_length:
                break
            max_freq = 0.0
            max_pair = None
            max_id = None
            if mode == 'mix':
                for t, pair in pairs.items():
                    freqs = pair.freqs
                    fo = freqs.get(idx, 0.0)
                    fi = freqs.get(-1, 0.0)
                    if fo > 0 or fi > 0:
                        freq = 10000 * fi + fo
                        if freq > max_freq:
                            max_freq = freq
                            max_pair = pair
                            max_id = t
            elif mode == 'input':
                for t, pair in pairs.items():
                    freqs = pair.freqs
                    freq = freqs.get(idx, 0.0)
                    if freq > 0:
                        if freq > max_freq:
                            max_freq = freq
                            max_pair = pair
                            max_id = t
            else:
                for t, pair in pairs.items():
                    freqs = pair.freqs
                    freq = freqs.get(idx, 0.0)
                    if freq > 0:
                        if freq > max_freq:
                            max_freq = freq
                            max_pair = pair
                            max_id = t
            if max_pair is None:
                break
            ids.append(max_id)
            pairs = max_pair.pairs
            length += 1

        return ids, np.tril(np.ones((length + 1, length + 1), dtype=np.int64), 0), [length]

    def _match(self, token_ids, mode='output', idx=-1):
        pairs = self.nodes
        token_id = None
        if len(token_ids) == 0:
            return token_id, pairs

        for token_id in token_ids:
            pair = pairs.get(token_id, None)
            pairs = {}
            if pair is None:
                break

            if mode == 'input':
                if pair.freqs.get(idx, 0.0) > 0:
                    pairs = pair.pairs
            elif mode == 'output':
                if pair.freqs.get(-1, 0.0) > 0:
                    pairs = pair.pairs
            else:
                if pair.freqs.get(idx, 0.0) > 0 or pair.freqs.get(-1, 0.0) > 0:
                    pairs = pair.pairs

        return token_id, pairs

    def _ravel(self, pairs, ids, mask, pid, max_size=64, max_length=8,
               min_output_freq=1.0, min_input_freq=1.0, min_mix_freq=1.0,
               output_weight=1e-4,
               sizes=None, mode='mix', idx=0):
        if len(ids) >= max_size or max_length <= 0:
            return

        sorts = [(k, v, (1.0 - output_weight) * v.freqs.get(idx, 0.0) + output_weight * v.freqs.get(-1, 0.0)) for k, v
                 in pairs.items()]
        sorts = sorted(sorts,
                       key=lambda x: x[2],
                       reverse=True)
        for tid, pair, fm in sorts:
            if len(ids) >= max_size:
                return
            fi = pair.freqs.get(idx, 0.0)
            fo = pair.freqs.get(-1, 0.0)
            # fm = (1 - output_weight) * fi + output_weight * fo
            if mode == 'mix':
                if fi < min_input_freq and fo < min_output_freq and fm < min_mix_freq:
                    continue
            elif mode == 'input':
                if fi < min_input_freq:
                    continue
            else:
                if fo < min_output_freq:
                    continue
            if fi > 0.0:
                sizes[0] += 1
            if fo > 0.0:
                sizes[1] += 1
            ids.append(tid)
            rid = len(ids) - 1

            if pid > -1:
                mask[rid] = mask[pid]
            mask[rid, rid] = 1
            if len(pair.pairs) > 0:
                self._ravel(pair.pairs, ids, mask, rid,
                            max_size=max_size,
                            max_length=max_length - 1,
                            min_output_freq=min_output_freq,
                            min_input_freq=min_input_freq,
                            min_mix_freq=min_mix_freq,
                            output_weight=output_weight,
                            sizes=sizes,
                            mode=mode,
                            idx=idx)

    def squeeze(self):
        if self.n_node > self.max_node or self.n_output_node > self.max_output_node:
            self._squeeze(self.nodes)
            sizes = [0]
            self._count_node(self.nodes, sizes)
            self.n_node = sizes[0]
            self.n_output_node = sizes[0]

    def _squeeze(self, pairs):
        for t, p in list(pairs.items()):
            fo = p.freqs.get(-1, 0.0)
            # TODO
            if fo > 1.0:
                p.freqs[-1] *= 0.5
                if len(p.pairs) > 0:
                    self._squeeze(p.pairs)
            else:
                pairs.pop(t)

    def _count_node(self, ps, sizes):
        l = len(ps)
        sizes[0] += l
        for t, p in ps.items():
            if len(p.pairs) > 0:
                self._count_node(p.pairs, sizes)

    def reset_input_freq(self):
        if len(self.nodes) == 0:
            return
        self._reset_input_freq(self.nodes)

    def _reset_input_freq(self, ps):
        for t, p in ps.items():
            freqs = p.freqs
            hit = False
            for idx, freq in list(freqs.items()):
                if idx >= 0 and freq > 0.0:
                    freqs[idx] = 0.0
                    hit = True
            if not hit:
                continue
            if len(p.pairs) > 0:
                self._reset_input_freq(p.pairs)


class LookaheadCache():
    def __init__(self, debug=False, eos=2, stop_words=None, max_node=512, max_output_node=256):
        self.debug = debug
        self.eos = eos
        self.max_node = max_node
        self.max_output_node = max_output_node
        self.mem = {}
        self._output_ids = defaultdict(list)
        self._update_trees = set()
        self._update_input_trees = set()
        self.stop_words = stop_words if stop_words is not None else {}
        self.default_mask = np.ones((1, 1), dtype=np.int64)

    def put(self, token_ids, branch_length=8, final=False, mode='output', idx=-1):
        if len(token_ids) >= 2:
            ts = len(token_ids)   # ts: token_ids size

            for i in range(ts - 1):
                token_id = token_ids[i]
                tup = token_ids[i + 1:i + branch_length + 1]
                if self.debug:
                    print(f'input token:{token_id} tokens:{tup}')
                tree = self.mem.get(token_id, None)
                if tree is not None:
                    tree.put(tup, mode=mode, idx=idx)
                else:
                    tree = Tree(token_id, max_node=self.max_node, max_output_node=self.max_output_node)
                    tree.put(tup, mode=mode, idx=idx)
                    self.mem[token_id] = tree
                self._update_trees.add(tree)
                if mode == 'input':
                    self._update_input_trees.add(tree)

        if final:
            self.reset_input_freqs()
            self.squeeze_branch_counts()

    def stream_put(self, token_ids, branch_length=8, final=False, mode='output', idx=0):
        # idx is only used for caching output_ids
        assert mode == 'output' and idx >= 0
        self._output_ids[idx].extend(token_ids)
        output_ids = self._output_ids[idx]
        ts = len(output_ids)
        min_branch_length = 1 if final else branch_length
        if ts > min_branch_length:
            for i in range(ts - min_branch_length):
                token_id = output_ids[i]
                tup = output_ids[i + 1:i + branch_length + 1]
                if self.debug:
                    print(f'input token:{token_id} tokens:{tup}')
                tree = self.mem.get(token_id, None)
                if tree:
                    tree.put(tup, mode='output', idx=-1)
                else:
                    tree = Tree(token_id, max_node=self.max_node, max_output_node=self.max_output_node)
                    tree.put(tup, mode='output', idx=-1)
                    self.mem[token_id] = tree
                self._update_trees.add(tree)
            if not final:
                self._output_ids[idx] = output_ids[ts - branch_length:]
        if final:
            self._output_ids[idx] = []
            self.reset_input_freqs()
            self.squeeze_branch_counts()

    def hier_get(self, token_ids, decoding_length=64, branch_length=8, min_input_size=0, min_output_size=0, mode='mix',
                 idx=0):
        assert mode in ('input', 'output', 'mix')

        decoding_masks = self.default_mask
        if decoding_length <= 1 or branch_length <= 1:
            return token_ids[-1:], decoding_masks, []

        decoding_ids = None
        sizes = [0, 0]
        match_count = len(token_ids)
        for i, t in enumerate(token_ids):
            tree = self.mem.get(t, None)
            if tree is not None:
                ids = token_ids[i + 1:]
                if t in self.stop_words and len(ids) == 0:
                    continue
                decoding_ids, decoding_masks, sizes = tree.get(ids,
                                                                          max_size=decoding_length,
                                                                          max_length=branch_length,
                                                                          min_input_size=min_input_size,
                                                                          min_output_size=min_output_size,
                                                                          mode=mode,
                                                                          idx=idx)
                s = len(decoding_ids)
                match_count = len(token_ids) - i
                if s >= branch_length or self.eos in decoding_ids:
                    break

        if decoding_ids is None:
            decoding_ids = token_ids[-1:]

        return decoding_ids, decoding_masks, sizes

    def block_get(self, token_ids, decoding_length=16, branch_length=8, min_input_size=0, min_output_size=0, mode='mix',
                  idx=0):

        output_ids, decoding_masks, decoding_lengths = self.trie_get(token_ids,
                                                                     decoding_length=decoding_length,
                                                                     branch_length=branch_length,
                                                                     min_input_size=min_input_size,
                                                                     min_output_size=min_output_size,
                                                                     mode=mode,
                                                                     idx=idx)
        sets = []
        true_decoding_length = len(output_ids) - 1
        for i in range(true_decoding_length, 0, -1):
            indices, = np.nonzero(decoding_masks[i, 1:])
            indices = set(indices)
            flag = True
            for ss in sets:
                if len(indices - ss) == 0:
                    flag = False
                    break
            if flag:
                sets.append(indices)

        sets.reverse()
        count = 0
        max_decoding_length = true_decoding_length
        branches = []
        for indices in sets:
            indices = sorted(list(indices))
            rest_count = max_decoding_length - count
            indices = indices[:rest_count]
            count += len(indices)
            branch = []
            for i in indices:
                branch.append(output_ids[i + 1])
            branches.append(branch)
            if count >= max_decoding_length:
                break
        ids = [output_ids[0]]
        masks = np.tril(np.ones((count + 1, count + 1)), 0)
        count = 1
        for branch in branches:
            ids.extend(branch)
            length = len(branch)
            masks[count:count + length, 1:count] = 0
            count += length

        # ps = sum([len(x) for x in sets])
        # print(f'hier:{true_decoding_length} par:{ps}/{count-1} rate:{ps/max(true_decoding_length,1):.3f}')

        return ids, masks, [count - 1]

    def one_get(self, token_ids, decoding_length=64, branch_length=8, min_input_size=0, min_output_size=0, mode='mix',
                idx=0):
        assert mode in ('input', 'output', 'mix')

        decoding_masks = self.default_mask
        if decoding_length <= 1 or branch_length <= 1:
            return token_ids[-1:], decoding_masks, []

        decoding_ids = []
        sizes = [0, 0]
        for i, t in enumerate(token_ids):
            tree = self.mem.get(t, None)
            if tree is not None:
                ids = token_ids[i + 1:]
                if t in self.stop_words and len(ids) == 0:
                    continue
                decoding_ids, decoding_masks, sizes = tree.get_one_branch(ids,
                                                                                     max_length=branch_length,
                                                                                     mode=mode,
                                                                                     idx=idx)
                s = len(decoding_ids)
                # too few tokens, retrieve again
                if s >= branch_length // 2 or self.eos in decoding_ids:
                    break

        return decoding_ids, decoding_masks, sizes

    def bat_get(self, token_id_list, decoding_length=64, branch_length=8, decoding_cursors=None, mode='output',
                indices=None, decoding_mode='hier'):
        assert mode in ('input', 'output', 'mix')
        assert decoding_mode in ('hier', 'one')
        bs = len(token_id_list)
        assert bs == len(decoding_cursors) and bs == len(indices), f'{bs=} {len(decoding_cursors)=} {len(indices)=}'

        decoding_id_list = []
        decoding_mask_list = []
        size_list = []

        min_cur = min(decoding_cursors)
        max_cur = max(decoding_cursors)
        bs = len(decoding_cursors)
        for sub_idx, token_ids in enumerate(token_id_list):
            update_decoding_length = decoding_length // bs
            min_input_size = 0
            min_output_size = max(update_decoding_length // 2, 1)
            method_name = decoding_mode + '_get'
            decoding_ids, decoding_masks, sizes = getattr(self, method_name)(token_ids,
                                                                                        decoding_length=update_decoding_length,
                                                                                        branch_length=branch_length,
                                                                                        min_input_size=min_input_size,
                                                                                        min_output_size=min_output_size,
                                                                                        mode=mode,
                                                                                        idx=indices[sub_idx])
            decoding_id_list.append(decoding_ids)
            decoding_mask_list.append(decoding_masks)
            size_list.append(sizes)

        bs = len(token_id_list)
        max_size = max([len(x) for x in decoding_id_list])

        decoding_masks = np.zeros((bs, max_size, max_cur - min_cur + max_size), dtype=np.int64)
        for i, decoding_ids in enumerate(decoding_id_list):
            org_size = len(decoding_ids)
            gap = max_size - org_size
            if gap > 0:
                decoding_ids.extend([self.eos] * gap)
            cur = decoding_cursors[i]
            decoding_masks[i, :org_size, cur - min_cur:cur - min_cur + org_size] = decoding_mask_list[i]
            decoding_masks[i, :, :cur - min_cur + 1] = 1
        return decoding_id_list, decoding_masks, size_list

    def fresh(self):
        self.mem = {}

    def reset_input_freqs(self):
        if len(self._update_input_trees) > 0:
            for c in self._update_input_trees:
                c.reset_input_freq()
            self._update_input_trees.clear()

    def squeeze_branch_counts(self):
        if len(self._update_trees) >= 1024:
            for c in self._update_trees:
                c.squeeze()
            self._update_trees.clear()

    def save_mem(self, save_mem_dir):
        cache_mem = self.mem
        serialized_object = pickle.dumps(cache_mem)
        json_string = json.dumps(serialized_object.decode('latin-1'))
        with open(save_mem_dir, 'w') as f:
            json.dump(json_string, f)

    def load_mem(self, load_mem_dir):
        with open(load_mem_dir, 'r') as f:
            json_string = json.load(f)
        deserialized_object = pickle.loads(json.loads(json_string).encode('latin-1'))
        cache_mem = deserialized_object
        self.mem = cache_mem
