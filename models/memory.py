#!/usr/bin/env python3

import torch
import torch.nn as nn


class Memory(nn.Module):
    eps = 2.5e-2
    momentum = 0.9
    chunk_size = 100000
    def __init__(self, N=500000, F=256):
        super().__init__()
        self.points = nn.Parameter(torch.FloatTensor(N, 3).fill_(1e7))
        self.descriptors = torch.zeros(N, F)
        self.register_buffer('rank', torch.zeros(N))
        self.cosine = PairwiseCosine() 
        self.N = N

    @torch.no_grad()
    def neighbor_search(self, points, radius):
        nei_idx = []
        chunk_start = 0
        for chunk in torch.split(self.points, self.chunk_size):
            _, c_idx = (torch.cdist(points, chunk) < radius).nonzero(as_tuple=True)
            nei_idx.append(c_idx + chunk_start)
            chunk_start += len(chunk)
        nei_idx = torch.cat(nei_idx)
        if nei_idx.numel() > 0:
            nei_idx = nei_idx.unique()
        return nei_idx, self.points[nei_idx]

    @torch.no_grad()
    def get_descriptors(self, idx):
        return self.descriptors[idx].to(idx.device)

    @torch.no_grad()
    def store(self, points, descriptors):
        idx = self.rank.topk(k=len(points), largest=False).indices
        self.descriptors[idx] = descriptors.to(self.descriptors)
        self.points[idx] = points
        self.rank[idx] = 1



    @torch.no_grad()
    def read(self, descriptors):
        cosine, idx = self.address(descriptors)
        return cosine, self.descriptors[idx].to(descriptors)

    @torch.no_grad()
    def update(self, idx, descriptors, loss):
        self.rank *= 0.999
        self.rank[idx] = self.moving(self.rank[idx], loss, 0.5)
        self.descriptors[idx] = self.moving(self.descriptors[idx], descriptors.to('cpu'), self.momentum)

    @torch.no_grad()
    def moving(self, x, y, momentum):
        return x * momentum + (1 - momentum) * y

    @torch.no_grad()
    def address(self, descriptors):
        mask = self.rank > 0
        cosine, idx = self.cosine(descriptors, self.descriptors[mask].to(descriptors)).max(dim=-1)
        return cosine, torch.arange(self.N).to(descriptors)[mask][idx]


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
    from tool import Timer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cuda:0, or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()

    N, F = 500, 256
    timer = Timer()
    memory = Memory().to(args.device)
    points = torch.randn(N, 3).to(args.device)
    descriptors = torch.randn(N, F).to(args.device)
    for i in range(100):
        memory.write(points, descriptors)
        cosine, descriptor = memory.read(descriptors)
        timer.toc()
