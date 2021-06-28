#!/usr/bin/env python3

import os
import bz2
import glob
import torch
import pickle
import pathlib
import numpy as np
from os import path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from utils.geometry import pose2mat

from .augment import AirAugment
from .base import DatasetBase


class TartanAir(DatasetBase):
    NED2EDN = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def __init__(self, root, scale=1, augment=True, catalog_dir=None):
        super().__init__(pathlib.Path(root) / 'tartanair', 'tartanair', catalog_dir)
        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = 320, 320, 320, 240
        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        self.augment = AirAugment(scale, size=[480, 640], resize_only=not augment)

    def _populate(self):
        for env_path in sorted(self.path_prefix.glob('*')):
            env = env_path.stem
            self.seqs[env] = [tuple(p.parts[-2:]) for p in sorted(env_path.glob('[EH]a[sr][yd]/*'))]

        self.poses, self.size = {}, {}
        for env, seq in self.get_env_seqs():
            seq_path = self.path_prefix / env / seq[0] / seq[1]

            pose_q = np.loadtxt(seq_path / 'pose_left.txt', dtype=np.float32)
            self.poses[env, seq] = self.NED2EDN @ pose2mat(pose_q)

            self.size[env, seq] = len([p for p in os.listdir(seq_path / 'image_left/') if p.endswith('.png')])

        return ['poses', 'size']

    def get_size(self, env, seq):
        return self.size[env, seq]

    def getitem_impl(self, env, seq, idx):
        seq_path = self.path_prefix / env / seq[0] / seq[1]
        image = Image.open(seq_path / 'image_left' / ('%0.6d_left.png' % idx))
        depth = F.to_pil_image(np.load(seq_path / 'depth_left' / ('%0.6d_left_depth.npy' % idx)), mode='F')
        pose = self.poses[env, seq][idx]
        image, K, depth = self.augment(image, self.K, depth)
        return image, (depth, pose, K)

    def summary(self):
        return pd.DataFrame(data=[
            [env, seq[0], seq[1], self.get_size(env, seq)] for env, seq in self.get_env_seqs()], 
            columns=['env', 'dif', 'id', 'size'])


class TartanAirTest(Dataset):
    def __init__(self, root, scale=1, augment=False, catalog_path=None):
        super().__init__()
        self.augment = AirAugment(scale, size=[480, 640], resize_only=not augment)
        if catalog_path is not None and os.path.exists(catalog_path):
            with bz2.BZ2File(catalog_path, 'rb') as f:
                self.sequences, self.image, self.poses, self.sizes = pickle.load(f)
        else:
            self.sequences = sorted(glob.glob(os.path.join(root,'mono','*')))
            self.pose_file = sorted(glob.glob(os.path.join(root,'mono_gt','*.txt')))
            self.image, self.poses, self.sizes = {}, {}, []
            ned2den = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            for seq, pose in zip(self.sequences, self.pose_file):
                quaternion = np.loadtxt(pose, dtype=np.float32)
                self.poses[seq] = ned2den @ pose2mat(quaternion)
                self.image[seq] = sorted(glob.glob(path.join(seq, '*.png')))
                assert(len(self.image[seq])==self.poses[seq].shape[0])
                self.sizes.append(len(self.image[seq]))
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with bz2.BZ2File(catalog_path, 'wb') as f:
                pickle.dump((self.sequences, self.image, self.poses, self.sizes), f)
        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = 320, 320, 320, 240
        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, ret):
        i, frame = ret
        seq, K = self.sequences[i], self.K
        image = Image.open(self.image[seq][frame])
        pose = self.poses[seq][frame]
        image, K = self.augment(image, self.K)
        return image, pose, K


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T

    data = TartanAir('/data/datasets/tartanair', scale=1, augment=True)
    sampler = AirSampler(data, batch_size=4, shuffle=True)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=4, pin_memory=True)

    test_data = TartanAirTest('/data/datasets/tartanair_test', scale=1, augment=True)
    test_sampler = AirSampler(test_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    for i, (image, depth, pose, K) in enumerate(loader):
        print(i, image.shape, depth.shape, pose.shape, K.shape)

    for i, (image, pose, K) in enumerate(test_loader):
        print(i, image.shape, pose.shape, K.shape)
