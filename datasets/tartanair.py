#!/usr/bin/env python3

import torch
from torchvision.datasets import VisionDataset


class TartanAir(VisionDataset):
    def __init__(self, root='/data/', train=True, download=None):
        super().__init__(root)
        self.train = train

    def __len__(self):
        return 100

    def __getitem__(self, index):
        image = torch.randn(3,320,320)
        depth = torch.randn(1,320,320)
        pose = torch.randn(3,4)
        if self.train is True:
            return image, depth, pose
        else:
            return image