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

        self.tMP = torch.nn.Parameter(torch.tensor(MPs[:,1:], dtype=torch.double)).requires_grad_()
        self.tKF = torch.nn.Parameter(torch.reshape(torch.tensor(KFs[:,1:], dtype=torch.double),(KFs.shape[0],4,4))).requires_grad_()

        # indexing the matches of key frames and Map Point
        self.idxMP = torch.tensor(MPs[:,0], dtype=torch.int)
        self.idxKF = torch.tensor(KFs[:,0], dtype=torch.int)

        # change the point to homogeneous
        self.tMPhomo = torch.unsqueeze(F.pad(input=self.tMP, pad=(0, 1), mode='constant', value=1), 2)

        #new pytorch provides inverse function for batch
        self.invtKF = torch.inverse(self.tKF)

    def forward(self, measurements):
        #indexing ## it is the same wiht torch where TODO
        indexKF = torch.where(torch.tensor(measurements[:,0].reshape(-1,1),dtype=torch.int)==self.idxKF)[1]
        indexMP = torch.where(torch.tensor(measurements[:,1].reshape(-1,1),dtype=torch.int)==self.idxMP)[1]

        # In test set, tKF is the Twc inverse. We may need to store something else in Memory
        self.reprojectPoints = torch.matmul(self.tKF[indexKF],self.tMPhomo[indexMP])

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

    parser = argparse.ArgumentParser(description='Test BAGD')
    parser.add_argument("--device", type=str, default='cuda', help="cuda, cuda:0, or cpu")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.000001, help='Random seed.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    MPs = np.loadtxt("../data/BAtest/MP.txt")
    KFs = np.loadtxt("../data/BAtest/KF.txt")
    Matches = np.loadtxt("../data/BAtest/Match.txt")

    net = BAGDnet(MPs, KFs)

    SmoothLoss = nn.SmoothL1Loss(beta = math.sqrt(5.99))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    z = torch.tensor(Matches[:,2:4])
    for i in range(20):
        output = net.forward(Matches)
        loss = SmoothLoss(output,z)
        loss.backward()
        optimizer.step()
        print(loss.data.item())
