#!/usr/bin/env python3

import os
import glob
import torch
import numpy as np
import kornia as kn
from os import path
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import functional as F

from .augment import AirAugment


class TartanAir(Dataset):
    def __init__(self, root, scale=1, augment=True):
        super().__init__()
        self.augment = augment if augment is None else AirAugment(scale, size=[480, 640])
        self.sequences = glob.glob(os.path.join(root,'*','Easy','*'))
        self.image, self.depth, self.poses, self.sizes = {}, {}, {}, []
        ned2den = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        for seq in self.sequences:
            quaternion = np.loadtxt(path.join(seq, 'pose_left.txt'), dtype=np.float32)
            self.poses[seq] = ned2den @ pose2mat(quaternion)
            self.image[seq] = sorted(glob.glob(path.join(seq,'image_left','*.png')))
            self.depth[seq] = sorted(glob.glob(path.join(seq,'depth_left','*.npy')))
            assert(len(self.image[seq])==len(self.depth[seq])==self.poses[seq].shape[0])
            self.sizes.append(len(self.image[seq]))

        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = 320, 320, 320, 240
        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, ret):
        i, frame = ret
        seq, K = self.sequences[i], self.K
        image = Image.open(self.image[seq][frame])
        depth = F.to_pil_image(np.load(self.depth[seq][frame]), mode='F')
        pose = self.poses[seq][frame]
        if self.augment is not False:
            image, K, depth = self.augment(image, self.K, depth)
        return image, depth, pose, K


class TartanAirTest(TartanAir):
    def __init__(self, root, scale=1, augment=True):
        super().__init__(root, scale, augment)
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

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, ret):
        i, frame = ret
        seq, K = self.sequences[i], self.K
        image = Image.open(self.image[seq][frame])
        pose = self.poses[seq][frame]
        if self.augment is not False:
            image, K = self.augment(image, self.K)
        return image, pose, K


class AirSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=True):
        self.data_sizes = data.sizes
        self.bs = batch_size
        self.shuffle = shuffle
        self.__iter__()

    def __iter__(self):
        batches = []
        for i, size in enumerate(self.data_sizes):
            num = size - self.bs + 1
            L = torch.randperm(num) if self.shuffle else torch.arange(num)
            batches += [list(zip([i]*self.bs, range(L[n], L[n]+self.bs))) for n in range(num)]
        L = torch.randperm(len(batches))
        self.batches = [batches[L[n]] for n in range(len(batches))] if self.shuffle else batches
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def pose2mat(pose):
    """Converts pose vectors to matrices.
    Args:
      pose: [tx, ty, tz, qx, qy, qz, qw] (N, 7).
    Returns:
      [R t] (N, 3, 4).
    """
    t = pose[:, 0:3, None]
    rot = R.from_quat(pose[:, 3:7]).as_matrix().astype(np.float32).transpose(0, 2, 1)
    t = -rot @ t
    return torch.cat([torch.from_numpy(rot), torch.from_numpy(t)], dim=2)


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
