#!/usr/bin/env python3

import os
import torch
import numpy as np
import torch.nn as nn
from utils.misc import rectify_savepath


class Memory(nn.Module):
    MAX_CAP = 2000
    STATE_DICT = ['n_probe', 'swap_dir', 'out_device', 'capacity',
                  'n_frame', 'property_spec', '_states', '_store', '_rel']

    def __init__(self, capacity=MAX_CAP, n_probe=1200, img_size=(240, 320), swap_dir='./memory', out_device='cuda'):
        super().__init__()
        self.n_probe = n_probe
        self.swap_dir = swap_dir
        self.out_device = out_device
        self.capacity = capacity
        self.n_frame, self.n_frame_hist = 0, 0
        self.property_spec = {
            'pos': {'shape': (n_probe, 3), 'default': np.nan, 'device': 'cuda'},
            'img': {'shape': (3,) + img_size, 'default': np.nan},
            'pose': {'shape': (3, 4), 'default': np.nan},
            'K': {'shape': (3, 3), 'default': np.nan},
            'depth_map': {'shape': (1,) + img_size, 'default': np.nan},
        }
        self._states = {}
        self._store = None
        self._rel = None

    # exp_rand sample
    # def sample_frames(self, n):
    #     n_frame = len(self._store)
    #     lamb = n_frame / 20
    #     prob = torch.exp((torch.arange(n_frame, dtype=torch.float32) - n_frame) / lamb)
    #     frame_idx = torch.multinomial(prob, n, replacement=(n_frame < n))
    #     # frame_idx = torch.arange(n_frame - n, n_frame)
    #     return [item.to(self.out_device) for item in self._store[frame_idx]]
    ################################################################

    # ################################################################ relevance sample
    # def sample_frames(self, n, project):
    #     fd, pd, pos = self._store[torch.arange(len(self._store))]
    #     relevance, mem_pts_scr = project(pos.to(self.out_device))
    #     relevance = relevance.T
    #     mem_pts_scr = mem_pts_scr.transpose(0, 1)
    #     # vis
    #     # sample
    #     cutoff = relevance.where(relevance > 0, torch.tensor(np.nan).to(relevance)).nanquantile(torch.tensor([0.1, 0.9]).to(relevance), dim=1, keepdim=True)
    #     if cutoff.isnan().all():
    #         return [None] * 7
    #     pos_prob = (relevance >= cutoff[1].clamp(max=0.4)).to(torch.float)
    #     neg_prob = (relevance <= cutoff[0]).to(torch.float)
    #     pos_idx = torch.multinomial(pos_prob, n, replacement=True)
    #     neg_idx = torch.multinomial(neg_prob, n, replacement=True)

    #     mem_pts_scr_offset = pos_idx + torch.arange(len(relevance)).unsqueeze(1).to(pos_idx) * len(self._store)
    #     return pd[pos_idx.cpu()].to(self.out_device), \
    #         pd[neg_idx.cpu()].to(self.out_device), \
    #         mem_pts_scr.reshape(-1, *mem_pts_scr.shape[2:])[mem_pts_scr_offset], \
    #         relevance.reshape(-1, *relevance.shape[2:])[mem_pts_scr_offset], \
    #         relevance.reshape(-1, *relevance.shape[2:])[neg_idx + torch.arange(len(relevance)).unsqueeze(1).to(neg_idx) * len(self._store)], \
    #         pos_idx, neg_idx
    # ################################################################
    # augmented sample
    def sample_frames(self, n_anchor, n_recent=0, n_pair=1, n_try=5):
        n_frame = len(self._store)
        for _ in range(n_try):
            # combination of most recent frames and random frames prior to those
            ank_idx = torch.cat([
                torch.randint(n_frame - n_recent, (n_anchor - n_recent,)),
                torch.arange(n_frame - n_recent, n_frame)])
            relevance = self._rel[ank_idx, :self.n_frame]
            # sample
            cutoff = relevance.where(
                relevance > 0, torch.tensor(np.nan).to(relevance)).nanquantile(
                torch.tensor([0.1, 0.9]).to(relevance),
                dim=1, keepdim=True)
            if cutoff.isfinite().all():
                break
        else:
            return [[None] * 3] * 2 + [[None] * 2]

        pos_prob = (relevance >= cutoff[1].clamp(max=0.4)).to(torch.float)
        neg_prob = (relevance <= cutoff[0]).to(torch.float)
        pos_idx = torch.multinomial(pos_prob, n_pair, replacement=True)
        neg_idx = torch.multinomial(neg_prob, n_pair, replacement=True)

        ank_batch = self._store[ank_idx]
        pos_batch = self._store[pos_idx]
        neg_batch = self._store[neg_idx]

        return (ank_idx, pos_idx, neg_idx), \
            (ank_batch, pos_batch, neg_batch), \
            (relevance.gather(1, pos_idx), relevance.gather(1, neg_idx))
            ################################################################

    def store_fifo(self, pos, proj_fn, **properties):
        frame_addr = (torch.arange(len(pos)) + self.n_frame_hist) % self.capacity
        self.n_frame_hist += len(frame_addr)
        self._store.store(frame_addr, pos=pos, **properties)
        self.n_frame = len(self._store)
        self.update_rel(frame_addr, proj_fn)

    def swap(self, name):
        if name in self._states:
            self._store, self._rel = self._states[name]
        else:
            self._store = SparseStore(name=name, out_device=self.out_device, max_cap=self.capacity, **self.property_spec)
            self._rel = torch.zeros(self.capacity, self.capacity, device=self.out_device).fill_(np.nan)
            self._states[name] = self._store, self._rel
        self.n_frame = len(self._store)
        #     else:
        #         torch.save(self._store, os.path.join(self.swap_dir, '%s.pth' % self._store.name))
        # load_path = os.path.join(self.swap_dir, '%s.pth' % name)
        # self._store = torch.load(load_path) if os.path.isfile(load_path) else \
        #     KeyFrameStore(self.D_point, self.D_frame, self.n_fea, name).to(device)
    
    def update_rel(self, frame_idx, proj_fn):
        old_pos = self._store[torch.arange(self.n_frame), ['pos']]['pos'].to(self.out_device)
        relevance = proj_fn(old_pos) # (n_frame, B)
        self._rel[:self.n_frame, frame_idx.to(self._rel.device)] = relevance
        self._rel[frame_idx.to(self._rel.device), :self.n_frame] = relevance.T

    # def get_rel(self, idx):
    #     # i0, i1, ..., i(i-1), ii, (i+1)i ..., ni
    #     return torch.stack([torch.cat([self._rel[i, :i + 1], self._rel[i + 1:self.n_frame, i].T]) for i in idx])
    def save(self, path, overwrite=True):
        save_path = rectify_savepath(path, overwrite=overwrite)
        torch.save(self, save_path)
        print('Saved memory: %s' % save_path)

    
    def load(self, path):
        loaded_mem = torch.load(path)
        for attr in self.STATE_DICT:
            setattr(self, attr, getattr(loaded_mem, attr))
        print('Loaded memory: %s' % path)

    def __len__(self):
        return self._store.__len__()

    def envs(self):
        return self._states.keys()


class SparseStore():

    def __init__(self, name='store', max_cap=2000, device='cpu', out_device='cuda', **property_spec):
        super().__init__()
        self.name = name
        self.buf = {}
        for name, specs in property_spec.items():
            shape = specs['shape']
            cap = specs.get('max_cap', max_cap)
            dtype = specs.get('dtype', torch.float32)
            dev = specs.get('device', device)
            def_val = specs.get('default', 0)

            self.buf[name] = {
                'shape': shape, 'capacity': cap, 'dtype': dtype, 'device': dev, 'default': def_val, 'values': {}}

        self.size = 0
        self.out_device = out_device

    @torch.no_grad()
    def store(self, _idx=None, **values):
        # sanity check
        batch = []
        for name, val in values.items():
            prop_shape = self.buf[name]['shape']
            assert prop_shape == val.shape[-len(prop_shape):]
            batch.append(val.shape[:-len(prop_shape)])
        assert all(b == batch[0] for b in batch)

        # single or batch mode
        if len(batch[0]) == 0:
            _idx = torch.tensor(_idx if _idx is not None else self.size, dtype=torch.long)
        else:
            batch_size = int(batch[0][0])
            _idx = torch.tensor(_idx, dtype=torch.long) if _idx is not None else torch.arange(batch_size) + self.size
            assert batch_size == len(_idx)

        for name, val in values.items():
            self._store(self.buf[name], _idx, val)

        self.size = max(len(buf['values']) for buf in self.buf.values())

    def _store(self, buf, idx, value):
        value = value.to(buf['device'])
        if len(idx.shape) == 0:
            buf['values'][int(idx)] = value
        else:
            for i, val in zip(idx.tolist(), value.to(buf['device'], non_blocking=True)):
                buf['values'][int(i)] = val

    def _get(self, buf, idx):
        if int(idx) in buf['values']:
            return buf['values'][int(idx)]
        else:
            return torch.zeros(buf['shape'], dtype=buf['dtype'], device=buf['device']).fill_(buf['default'])

    @torch.no_grad()
    def __getitem__(self, idx):
        idx, include = idx if isinstance(idx, tuple) else (idx, self.buf.keys())
        idx, ret = torch.tensor(idx), {name: [] for name in self.buf if name in include}
        if len(idx.shape) == 0:
            for name, buf in self.buf.items():
                ret[name].append(self._get(buf, idx))
            return {name: tensors[0] for name, tensors in ret.items()}
        else:
            for name in ret:
                for i in idx.flatten():
                    ret[name].append(self._get(self.buf[name], i))
            # make returned tensor respect shape of index
            return {name: torch.stack(tensors).reshape(*idx.shape, *tensors[0].shape).to(self.out_device, non_blocking=True)
                    for name, tensors in ret.items()}

    def __len__(self):
        return self.size


def test_store():
    store = SparseStore(pos={'shape': (12, 3), 'device': 'cuda', 'default': np.nan}, idx={'shape': (1,), 'dtype': torch.long})
    pos = torch.arange(360).reshape(10, 12, 3).to(torch.float)
    idx = torch.arange(10).reshape(10, 1)
    store.store(0, pos=pos[0], idx=idx[0])
    store.store(pos=pos[1:3], idx=idx[1:3])
    store.store(torch.arange(1, 4).cuda(), pos=pos[4:7], idx=idx[4:7])
    print(len(store))
    print(store[1])
    print(store[[0, 2]])
    print(store[[[0], [4]]])


if __name__ == '__main__':
    test_store()
