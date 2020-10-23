#!/usr/bin/env python3

import torch
from torchvision.datasets import VisionDataset

import numpy as np

from models.utils import pose2mat
from .datasetbase import DatasetBase


class TartanAir(DatasetBase):
    def __init__(self, root='/data/',
                 img_dir='image_left', img_transform=None,
                 depth_dir='depth_left', depth_transform=None,
                 pose_file='pose_left.txt', train=True, download=None):
        super().__init__(root, img_dir=img_dir, img_transform=img_transform,
                         depth_dir=depth_dir, depth_transform=depth_transform,
                         pose_file=pose_file,
                         train=train, download=download)

    def load_poses(self, pose_file):
        poses7 = np.loadtxt(pose_file).astype(np.float32)
        assert(poses7.shape == (self.size, 7))  # position + quaternion
        return pose2mat(poses7)
