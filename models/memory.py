#!/usr/bin/env python3

import torch
import torch.nn as nn


class Memory(nn.Module):
    eps = 1e-3
    momentum = 0.999
    def __init__(self, N=500000, F=256):
        super().__init__()
        self.points = nn.Parameter(torch.FloatTensor(N, 3).fill_(1e7))
        self.register_buffer('descriptors', torch.zeros(N, F))
        self.register_buffer('usage',  torch.zeros(N, dtype=torch.int32))
        self.register_buffer('idx',  torch.arange(N))
        self.cosine = PairwiseCosine() 

    @torch.no_grad()
    def write(self, points, descriptors):
        idx, momentum = self.point_address(points)
        self.usage[idx] += 1
        self.points[idx] = self.moving(self.points[idx], points, momentum)
        self.descriptors[idx] = self.moving(self.descriptors[idx], descriptors, momentum)
        return self.descriptors[idx]

    @torch.no_grad()
    def read(self, descriptors):
        cosine, idx = self.address(descriptors)
        return cosine, self.descriptors[idx]

    @torch.no_grad()
    def moving(self, x, y, momentum):
        return x * momentum + (1 - momentum) * y

    @torch.no_grad()
    def point_address(self, points):
        dist, idx = torch.cdist(points, self.points, p=2).min(dim=-1)
        mask = dist > self.eps
        idx[mask] = self.usage.topk(k=mask.sum(), largest=False).indices
        momentum = torch.zeros_like(idx)
        momentum[mask == 0] = self.momentum
        return idx, momentum.unsqueeze(-1)

    @torch.no_grad()
    def address(self, descriptors):
        mask = self.usage > 0
        cosine, idx = self.cosine(descriptors, self.descriptors[mask]).max(dim=-1)
        return cosine, self.idx[mask][idx]


class PairwiseCosine(nn.Module):
    def __init__(self, dim=-1, eps=1e-7):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.eqn = 'md,nd->mn'

    def forward(self, x, y):
        xx = x.norm(dim=self.dim).unsqueeze(-1)
        yy = y.norm(dim=self.dim).unsqueeze(-2)
        xy = torch.einsum('md,nd->mn', x, y)
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
