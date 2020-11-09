#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F


class BAGDnet(nn.Module):
    def __init__(self, MPs, KFs):
        super().__init__()
        self.fx = 320
        self.fy = 320
        self.cx = 320
        self.cy = 240

        self.tMP = nn.Parameter(MPs[:,1:] + torch.randn_like(MPs[:,1:])*0.001)
        self.tKF = nn.Parameter(KFs[:,1:].view(-1,4,4))

        # indexing the matches of key frames and Map Point
        self.idxMP = MPs[:,0].type(torch.int)
        self.idxKF = KFs[:,0].type(torch.int)

        # change the point to homogeneous
        self.tMPhomo = F.pad(input=self.tMP, pad=(0, 1), mode='constant', value=1).unsqueeze(2)

        #new pytorch provides inverse function for batch
        self.invtKF = self.tKF.inverse()

    def forward(self, measurements):
        #indexing ## it is the same wiht torch where TODO
        indexKF = torch.where(measurements[:,0].reshape(-1,1).type(torch.int)==self.idxKF)[1]
        indexMP = torch.where(measurements[:,1].reshape(-1,1).type(torch.int)==self.idxMP)[1]

        # In test set, tKF is the Twc inverse. We may need to store something else in Memory
        self.reprojectPoints = self.tKF[indexKF] @ self.tMPhomo[indexMP]

        # This operation can also be a matmul with matrxi K TODO
        self.ptx = (self.reprojectPoints[:,0]/self.reprojectPoints[:,2])*self.fx + self.cx
        self.pty = (self.reprojectPoints[:,1]/self.reprojectPoints[:,2])*self.fy + self.cy

        obs2d = torch.cat((self.ptx,self.pty),1)
        return obs2d


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

    MPs = torch.from_numpy(np.loadtxt("data/BAtest/MP.txt"))
    KFs = torch.from_numpy(np.loadtxt("data/BAtest/KF.txt"))
    Matches = torch.from_numpy(np.loadtxt("data/BAtest/Match.txt"))

    net = BAGDnet(MPs, KFs)

    SmoothLoss = nn.SmoothL1Loss(beta = math.sqrt(5.99))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    z = Matches[:,2:4]
    for i in range(200):
        output = net(Matches)
        loss = SmoothLoss(output,z)
        loss.backward()
        optimizer.step()
        print('Epoch: %d, Loss: %.7f'%(i, loss))
        if scheduler.step(loss):
            print('Early Stopping!')
            break
