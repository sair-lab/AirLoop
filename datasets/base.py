#!/usr/bin/env python3

import bz2
import itertools
import re
import pathlib
import pickle
from copy import copy

import numpy as np

from torch.utils.data import Dataset, Sampler


class DatasetBase(Dataset):
    def __init__(self, root, name, catalog_dir=None):
        self.path_prefix = pathlib.Path(root)
        self._seqs = {}

        # load state dict if available
        if catalog_dir is None:
            self._populate()
        else:
            catalog_path = pathlib.Path(catalog_dir) / ('%s.pbz2' % name)
            if catalog_path.is_file():
                with bz2.BZ2File(catalog_path, 'rb') as f:
                    state_dict = pickle.load(f)
                    for k, v in state_dict.items():
                        setattr(self, k, v)
                print('Loaded catalog %s' % catalog_path)
            else:
                catalog_attr = ['_seqs'] + self._populate()
                state_dict = {attr: getattr(self, attr) for attr in catalog_attr}

                catalog_path.parent.mkdir(parents=True, exist_ok=True)
                with bz2.BZ2File(catalog_path, 'wb') as f:
                    pickle.dump(state_dict, f)

    @property
    def envs(self):
        return self._seqs.keys()

    @property
    def seqs(self):
        return self._seqs

    def _populate(self):
        raise NotImplementedError()

    def get_size(self, env, seq):
        raise NotImplementedError()

    def getitem_impl(self, env, seq, idx):
        raise NotImplementedError()

    def get_seq_id(self, env, seq):
        env, seq = np.atleast_1d(env).tolist(), np.atleast_1d(seq).tolist()
        return '_'.join(env + seq)

    def get_env_seqs(self):
        return [(env, seq) for env in self.envs for seq in self.seqs[env]]

    def __getitem__(self, index):
        env, seq, idx = index
        assert env in self.envs, 'No such environment: %s' % env
        assert seq in self.seqs[env], 'No such sequence in environment %s: %s' % (env, seq)
        assert 0 <= idx < self.get_size(env, seq), 'Index out of bound for (%s:%s): %d' % (env, seq, idx)
        item = self.getitem_impl(env, seq, idx)
        item = (item,) if not isinstance(item, tuple) else item
        return item + ((env, seq, idx),)

    def include_exclude(self, include=None, exclude=None):
        incl_pattern = re.compile(include) if include is not None else None
        excl_pattern = re.compile(exclude) if exclude is not None else None
        for env, seq in self.get_env_seqs():
            seq_id = self.get_seq_id(env, seq)
            if (incl_pattern and incl_pattern.search(seq_id) is None) or \
                    (excl_pattern and excl_pattern.search(seq_id) is not None):
                self.seqs[env].remove(seq)
                if not self.seqs[env]:
                    self.seqs.pop(env)

    def rand_split(self, ratio, seed=42):
        env_seqs = self.get_env_seqs()
        total, ratio = len(env_seqs), np.array(ratio)
        split_idx = np.cumsum(np.round(ratio / sum(ratio) * total), dtype=np.int)[:-1]
        subsets = []
        for perm in np.split(np.random.default_rng(seed=seed).permutation(total), split_idx):
            perm = sorted(perm)
            subset = copy(self)
            subset._seqs = {}
            for env, seq in np.take(np.array(env_seqs, dtype=object), perm, axis=0).tolist():
                subset.seqs.setdefault(env, []).append(seq)
            subsets.append(subset)
        return subsets


class DefaultSampler(Sampler):
    def __init__(self, dataset: DatasetBase, batch_size, seq_merge='shuffle', env_merge='rand', shuffle_batch=False, overlap=True):
        self.seq_sizes = [(env_seq, dataset.get_size(*env_seq)) for env_seq in dataset.get_env_seqs()]
        self.bs = batch_size
        self.batches = []
        env_batches = {}
        for env_seq, size in self.seq_sizes:
            frame_idx = np.arange(0, size)
            b_start = np.arange(0, size - self.bs, 1 if overlap else self.bs)
            batch = [[env_seq + (idx,) for idx in frame_idx[st:st+self.bs]] for st in b_start]
            if shuffle_batch:
                np.random.shuffle(batch)
            env_batches.setdefault(env_seq[0], []).extend(batch)

        if seq_merge == 'cat':
            pass
        elif seq_merge == 'rand_pick':
            for env, batches in env_batches.copy().items():
                env_samples = list(itertools.chain(*batches))
                np.random.shuffle(env_samples)
                # slice back into chunks
                env_batches[env] = [list(batch) for batch in itertools.zip_longest(*([iter(env_samples)] * batch_size))]
        
        if env_merge == 'cat':
            # A1 A2 A3 B1 B2 B3
            self.batches = list(itertools.chain(*env_batches.values()))
        elif env_merge == 'rand_interleave':
            # A1 B1 A2 B2 B3 A3
            selection = sum([[env] * len(batch) for i, (env, batch) in enumerate(env_batches.items())], [])
            np.random.shuffle(selection)
            self.batches = [env_batches[env].pop() for env in selection]
        elif env_merge == 'rand_pick':
            # B1 B2 A2 A1 A3 B3
            self.batches = list(itertools.chain(*env_batches.values()))
            np.random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
