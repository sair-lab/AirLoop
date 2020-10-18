#!/usr/bin/env python3

import torch
import torch.nn as nn

from collections import deque


class Memory(nn.Module):
    """
    A memory for storing (key, [record1, record2, ...]) pairs.
    """

    def __init__(self, key_shape,
                 merge_margin=0.05, similarity=nn.CosineSimilarity(),
                 drift_rate=0.01, normalizer=lambda x: 1.0,
                 init_cap=1000,
                 **attr_shapes_dict):
        """Creates storage files.

        Args:
          key_shape:        Shape of key.
          attr_shapes_dict: Shapes of each attribute {"attr_name": attr_shape}.
          merge_margin:     Merge two keys if cossim >= 1 - merge_margin.
          similarity:       Similarity measure.
          drift_rate:       Moving average weight of new key.
          normalizer:       Key scaling after moving average (K) -> float.
          init_cap:         Initial storage capacity. Grows on demand.
        """
        super(Memory, self).__init__()

        self.register_buffer('keys', torch.zeros((init_cap,) + key_shape))

        self.inodes = dict()
        self._inode_free = deque(range(init_cap))

        self.attr_names = attr_shapes_dict.keys()
        self._storage_free = deque(range(init_cap))
        for attr_name in attr_shapes_dict:
            self.register_buffer(attr_name, torch.zeros(
                (init_cap,) + attr_shapes_dict[attr_name]))

        self.drift_rate = drift_rate
        self.merge_margin = merge_margin
        self.similarity = similarity
        self.normalizer = normalizer

    def set_drift_rate(self, new_rate):
        """Sets key drift rate. Matched keys will be updated as:

        key = key * (1 - drift_rate) + new_key * drift_rate
        """
        self.drift_rate = new_rate

    def set_merge_margin(self, new_margin):
        """Sets key merging margin. Matched keys will merge if (cossim >= 1 - thresh)."""
        self.merge_margin = new_margin

    @torch.no_grad()
    def put(self, key, replace=False, **attr_dict):
        """Writes payload to memory and update matched (cossim >= 1 - merge_margin) keys.

        Args:
          key:       The key of shape (K).
          attr_dict: {"attr1": (attr1_shape), ...}.
          replace:   True if wants to overwrite the oldest record associated
                     with key.
        """
        key /= self.normalizer(key)
        matched_key_idx = self._match_key(key, 1 - self.merge_margin)
        if not matched_key_idx:
            # no match, make new inode
            inode_addr = self._alloc_inode()
            self.inodes[inode_addr] = []
            self.keys[inode_addr] = key
        else:
            inode_addr = self._merge_keys(matched_key_idx, key)

        inode = self.inodes[inode_addr]
        if replace and inode:
            self._storage_free.appendleft(inode.pop(0))

        # write to key and storage
        data_addr = self._alloc_storage()
        for attr in attr_dict:
            getattr(self, attr)[data_addr] = attr_dict[attr]

        inode.append(data_addr)

    @torch.no_grad()
    def get(self, key):
        """Fetch payload from all matched keys.

        Args:
          key: The key of shape (K).

        Returns:
          [(matched_key1, [{attr: data, ...}, ...]), ...]
        """
        key /= self.normalizer(key)
        matched_key_idx = self._match_key(key, 1 - self.merge_margin)
        results = []
        for addr in matched_key_idx:
            matched_key = self.keys[addr]
            payload = []
            for data_addr in self.inodes[addr]:
                payload.append(
                    {attr: getattr(self, attr)[data_addr] for attr in self.attr_names})
            results.append((matched_key, payload))

        return results

    def _match_key(self, key, thresh):
        """Return the indices of matched (cossim >= thresh) keys."""
        key_ = key.repeat(len(self.keys), 1)
        sim = self.similarity(key_, self.keys)
        return (sim >= thresh).nonzero().squeeze(1).tolist()

    def _merge_keys(self, key_indices, new_key):
        """Merge matched keys with new key.

        Average all old_keys, take moving average with new key and store in the
        location of old_keys[0]. Merge inodes and recycle empty key slot.

        Returns:
          Writing location for new_key.
        """
        new_key_addr = key_indices[0]
        old_key_addrs = key_indices[1:]
        self.keys[new_key_addr] = self.keys[key_indices].mean(dim=0) * (1 - self.drift_rate) \
            + new_key * self.drift_rate
        self.keys[new_key_addr] /= self.normalizer(self.keys[new_key_addr])

        self.keys[old_key_addrs] = 0
        for old_key_idx in old_key_addrs:
            self.inodes[new_key_addr].extend(self.inodes.pop(old_key_idx))
            self._inode_free.append(old_key_idx)

        return new_key_addr

    def _alloc_inode(self):
        """Get new key location, expand if necessary."""
        if not self._inode_free:
            self.keys = self._expand(self._inode_free, self.keys)
        return self._inode_free.popleft()

    def _alloc_storage(self):
        """Get new storage location, expand if necessary."""
        if not self._storage_free:
            for i, attr in enumerate(self.attr_names):
                setattr(self, attr, self._expand(
                    self._storage_free if i == 0 else None, getattr(self, attr)))
        return self._storage_free.popleft()

    def _expand(self, free_list, blob):
        """Double storage space, update free list."""
        if free_list is not None:
            cap = len(blob)
            free_list.extend(range(cap, cap * 2))
        return torch.cat([blob, torch.zeros_like(blob)])


if __name__ == "__main__":
    """Test"""
    import argparse

    parser = argparse.ArgumentParser(description='Test FeatureNet')
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cuda:0, or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = args.device

    def normalizer(x):
        return torch.norm(x, 2)

    mem = Memory((3,), a=(2,), b=(3,), init_cap=1,
                 drift_rate=0.5, normalizer=normalizer).to(device)

    keys = torch.rand(5, 3).to(device)
    keys[2] = keys[1] * 0.99 + keys[2] * 0.01
    a_data = torch.rand(5, 2).to(device)
    b_data = torch.rand(5, 3).to(device)

    mem.put(keys[0], a=a_data[0], b=b_data[0])
    mem.put(keys[0], a=a_data[1], b=b_data[1])
    mem.put(keys[1], a=a_data[3], b=b_data[3], replace=True)
    mem.put(keys[2], a=a_data[2], b=b_data[2], replace=True)

    # key0, 2 records
    query = keys[0] + torch.rand(3).to(device) * 0.1
    results = mem.get(query)
    assert len(results) == 1
    key, pay = results[0]
    assert len(pay) == 2
    assert torch.allclose(key, keys[0] / normalizer(keys[0]))
    for i in range(len(pay)):
        assert torch.allclose(pay[i]['a'], a_data[i])
        assert torch.allclose(pay[i]['b'], b_data[i])

    # key2, 1 record
    query = keys[1] + torch.rand(3).to(device) * 0.1
    results = mem.get(query)
    assert len(results) == 1
    key, pay = results[0]
    key_12 = keys[1] * 0.5 + keys[2] * 0.5
    assert torch.allclose(key, key_12 / normalizer(key_12))
    assert len(pay) == 1
    assert torch.allclose(pay[0]['a'], a_data[2])
    assert torch.allclose(pay[0]['b'], b_data[2])

    # merge all, 4 records
    mem.set_merge_margin(0.99)
    mem.put(keys[3], a=a_data[3], b=b_data[3])
    query = keys[3] + torch.rand(3).to(device) * 0.1
    results = mem.get(query)
    assert len(results) == 1
    key, pay = results[0]
    key0123 = keys[0] * 0.25 + key_12 * 0.25 + keys[3] * 0.5
    assert torch.allclose(key, key0123 / normalizer(key0123))
    assert len(pay) == 4
    for i in range(len(pay)):
        assert torch.allclose(pay[i]['a'], a_data[i])
        assert torch.allclose(pay[i]['b'], b_data[i])
