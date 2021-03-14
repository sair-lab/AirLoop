#!/usr/bin/env python3

import os
import torch
import numpy as np
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, D_point, D_frame, n_fea, swap_dir='./memory', out_device='cuda'):
        super().__init__()
        self.D_point, self.D_frame, self.n_fea = D_point, D_frame, n_fea
        self.swap_dir = swap_dir
        self.out_device = out_device
        self._store = None

    def sample_frames(self, n):
        n_frame = len(self._store)
        # frame_idx = torch.multinomial(torch.arange(n_frame, dtype=torch.float32), n, replacement=(n_frame < n))
        frame_idx = torch.arange(n_frame - n, n_frame)
        return [item.to(self.out_device) for item in self._store[frame_idx]]

    def store(self, frame_desc, point_desc, pos=None):
        self._store.store(frame_desc.cpu(), point_desc.cpu(), pos.cpu())

    def swap(self, name, device='cpu'):
        if self._store is not None:
            if name == self._store.name:
                return
            else:
                torch.save(self._store, os.path.join(self.swap_dir, '%s.pth' % self._store.name))
        load_path = os.path.join(self.swap_dir, '%s.pth' % name)
        self._store = torch.load(load_path) if os.path.isfile(load_path) else \
            KeyFrameStore(self.D_point, self.D_frame, self.n_fea, name).to(device)
        self._store = KeyFrameStore(self.D_point, self.D_frame, self.n_fea, name).to(device)
    
    def __len__(self):
        return self._store.__len__()

    def forward(self):
        raise NotImplementedError()


class KeyFrameStore(nn.Module):
    eps = 1e-3
    init_cap = 2000000
    init_kf_cap = 8000

    def __init__(self, D_point, D_frame, num_fea, name='store', merge_radius=1e-2):
        super().__init__()
        self.name = name
        self.n_frame, self.n_desc, self.n_point = 0, 0, 0
        self.D_point, self.D_frame = D_point, D_frame

        self.register_buffer('_fd', (torch.zeros(self.init_kf_cap, D_frame).fill_(np.nan)))
        self.register_buffer('_pd', torch.zeros(self.init_cap, D_point).fill_(np.nan))
        self.register_buffer('_kf_desc_idx', torch.LongTensor(self.init_kf_cap, num_fea).fill_(-1))

        self.merge_radius = merge_radius
        self.merge = False
        self._pos_kf_lookup = {}
        self.register_buffer('_pos', torch.zeros(self.init_cap, 3).fill_(np.inf))
        self.register_buffer('_kf_pos_idx', torch.LongTensor(self.init_kf_cap, num_fea).fill_(-1))

    @torch.no_grad()
    def store(self, frame_desc, point_desc, pos=None):
        fd_addr = torch.arange(self.n_frame, self.n_frame + len(frame_desc))
        self._fd[fd_addr] = frame_desc
        self.n_frame += len(fd_addr)

        self._store_desc(point_desc, fd_addr)
        if pos is not None:
            self._store_pos(pos, fd_addr)

    def _store_desc(self, pd, fd_addr):
        B, N, _ = pd.shape
        pd = pd.reshape(-1, self.D_point)
        pd_idx = torch.arange(self.n_desc, self.n_desc + len(pd)).to(self._kf_desc_idx)
        self._pd[pd_idx] = pd
        self.n_desc += len(pd)
        self._kf_desc_idx[fd_addr] = pd_idx.reshape(B, N)

    def _store_pos(self, pos, fd_addr):
        B, N, _ = pos.shape
        pos_addr = torch.LongTensor(B, N).fill_(-1).to(self._kf_pos_idx)
        pos = pos.reshape(B * N, 3)
        # merge points within radius
        merged_idx = None
        if self.merge:
            near_pairs = tc.radius(self._pos, pos, self.merge_radius)
            if near_pairs.numel() > 0:
                mem_addr, new_idx = near_pairs
                # make sure no two new points merge into the same mem point
                mem_addr, _, new_idx = self._groupby(mem_addr, new_idx)
                merged_idx, group_size, w_addr = self._groupby(new_idx, mem_addr)
                self._merge_points(mem_addr, group_size, w_addr, pos[merged_idx])
                pos_addr[merged_idx // N, merged_idx % N] = w_addr
        if merged_idx is None:
            merged_idx = torch.empty(0).to(pos_addr)
        # write unmerged points
        unmerged_idx = torch.tensor(list(set(np.arange(len(pos))) - set(merged_idx.tolist()))).to(pos_addr)
        w_addr = torch.arange(self.n_point, self.n_point + len(unmerged_idx)).to(pos_addr)
        self._pos[w_addr] = pos[unmerged_idx]
        self.n_point += len(unmerged_idx)
        pos_addr[unmerged_idx // N, unmerged_idx % N] = w_addr
        self._link_points(pos_addr, fd_addr)

    def _merge_points(self, src_addr, group_size, dst_addr, new_pos=None, alpha=0.9):
        updated_pos = self._pos[src_addr] * alpha + new_pos * (1 - alpha) \
            if new_pos is not None else self._pos[src_addr]
        # average within group and write back
        self._pos[dst_addr] = torch.cat([up.mean(dim=0, keep_dim=True) for up in updated_pos.split(group_size)])
        # combine keyframe references
        for sas, da in zip(src_addr.split(group_size), dst_addr.split(group_size)):
            self._pos_kf_lookup[da] = sum([self._pos_kf_lookup.pop(sa)
                                           for sa in sas.items() if sa != da], self._pos_kf_lookup[da])
            self._kf_pos_idx[list(zip(*self._pos_kf_lookup[da]))] = da

    def _link_points(self, pos_addr, fd_addr):
        self._kf_pos_idx[fd_addr] = pos_addr
        for pas, fa in zip(pos_addr.tolist(), fd_addr.tolist()):
            for pa in pas:
                self._pos_kf_lookup[pa] = [fa]

    @staticmethod
    def _groupby(x, *joint_select):
        val, cnt = x.unique_consecutive(return_counts=True)
        repr_idx = cnt.cumsum() - 1
        return (val, cnt) + tuple([js[repr_idx] for js in joint_select])

    def __getitem__(self, i):
        return self._fd[i], self._pd[self._kf_desc_idx[i]], self._pos[self._kf_pos_idx[i]]

    def __len__(self):
        return self.n_frame

    def forward(self):
        raise NotImplementedError()


class PairwiseCosine(nn.Module):
    def __init__(self, dim=-1, eps=1e-7):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.eqn = 'md,nd->mn'

    def forward(self, x, y):
        xx = x.norm(dim=self.dim).unsqueeze(-1)
        yy = y.norm(dim=self.dim).unsqueeze(-2)
        xy = torch.einsum(self.eqn, x, y)
        return xy / (xx * yy).clamp(min=self.eps)


if __name__ == "__main__":
    """Test"""
    B, N, D = 8, 250, 256
    kfs = KeyFrameStore(256, 256, 250).cuda()
    kfs.store(torch.randn(B, N, D).cuda(), torch.randn(B, D).cuda(), torch.randn(B, N, 3).cuda())
    kfs.store(torch.randn(B, N, D).cuda(), torch.randn(B, D).cuda(), torch.randn(B, N, 3).cuda())
