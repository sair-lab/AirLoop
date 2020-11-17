#!/usr/bin/env python3

import math
import torch
import numpy as np
import kornia as kn
import torch.nn as nn
from torchvision import models

class BAGDnet(nn.Module):
    def __init__(self, MPs, KFs, K):
        super().__init__()
        self.K = K
        self.tMP = nn.Parameter(MPs[:,1:])
        self.tKF = nn.Parameter(KFs[:,1:].view(-1,4,4))

        # indexing the matches of key frames and Map Point
        self.idxMP = MPs[:,0].type(torch.int)
        self.idxKF = KFs[:,0].type(torch.int)

        self.tMPhomo = kn.convert_points_to_homogeneous(self.tMP)

    def forward(self, frame_id, point_id):
        #indexing ## it is the same wiht torch where TODO
        indexKF = torch.where(frame_id==self.idxKF)[1]
        indexMP = torch.where(point_id==self.idxMP)[1]

        # In test set, tKF is the Twc inverse. We may need to store something else in Memory
        points = (self.tKF[indexKF] @ self.tMPhomo[indexMP].unsqueeze(-1)).squeeze(-1)

        Pc = kn.convert_points_from_homogeneous(points)

        return kn.project_points(Pc, self.K)


class ConsecutiveMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=-2)

    def forward(self, descriptors, points):
        desc_src = descriptors[:-1]
        desc_dst = descriptors[1:]
        # batch pairwise cosine
        x = desc_src.permute((1, 2, 0)).unsqueeze(1)
        y = desc_dst.permute((1, 2, 0))
        c = self.cosine(x, y)
        pcos = c.permute((2, 0, 1))

        confidence, idx = pcos.max(dim=2)
        matched = points[1:].gather(1, idx.unsqueeze(2).expand(-1, -1, 2))

        return matched, confidence


if __name__ == "__main__":
    '''Test codes'''
    import time, math
    import argparse
    import torch.optim as optim
    from tool import EarlyStopScheduler

    parser = argparse.ArgumentParser(description='Test BAGD')
    parser.add_argument("--device", type=str, default='cuda', help="cuda, cuda:0, or cpu")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Random seed.')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Random seed.')
    parser.add_argument("--factor", type=float, default=0.1, help="factor of lr")
    parser.add_argument("--patience", type=int, default=5, help="training patience")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    MPs = torch.from_numpy(np.loadtxt("data/BAtest/MP.txt")).cuda()
    KFs = torch.from_numpy(np.loadtxt("data/BAtest/KF.txt")).cuda()
    Matches = torch.from_numpy(np.loadtxt("data/BAtest/Match.txt")).cuda()

    fx, fy, cx, cy = 320, 320, 320, 240
    affine = torch.FloatTensor([[[fx, 0, cx], [0, fy, cy]]])
    K = kn.convert_affinematrix_to_homography(affine).cuda()

    net = BAGDnet(MPs, KFs, K)

    SmoothLoss = nn.SmoothL1Loss(beta = math.sqrt(5.99))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    pixel = Matches[:,2:4]
    frame_id = Matches[:,0,None].type(torch.int)
    point_id = Matches[:,1,None].type(torch.int)

    for i in range(200):
        output = net(frame_id, point_id)
        loss = SmoothLoss(output, pixel)
        loss.backward()
        optimizer.step()
        print('Epoch: %d, Loss: %.7f'%(i, loss))
        if scheduler.step(loss):
            print('Early Stopping!')
            break
