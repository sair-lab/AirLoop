#!/usr/bin/env python3

import os
import re
import bz2
import glob
import torch
import pickle
import numpy as np
import kornia as kn
from os import path
from PIL import Image
from copy import copy
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import functional as F

from .augment import AirAugment


class TartanAir(Dataset):
    def __init__(self, root, scale=1, augment=True, catalog_path=None, exclude=None, include=None):
        super().__init__()
        self.augment = AirAugment(scale, size=[480, 640], resize_only=not augment)
        if catalog_path is not None and os.path.exists(catalog_path):
            with bz2.BZ2File(catalog_path, 'rb') as f:
                self.sequences, self.image, self.depth, self.poses, self.sizes = pickle.load(f)
        else:
            self.sequences = glob.glob(os.path.join(root,'*','[EH]a[sr][yd]','*'))
            self.image, self.depth, self.poses, self.sizes = {}, {}, {}, []
            ned2den = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            for seq in self.sequences:
                quaternion = np.loadtxt(path.join(seq, 'pose_left.txt'), dtype=np.float32)
                self.poses[seq] = ned2den @ pose2mat(quaternion)
                self.image[seq] = sorted(glob.glob(path.join(seq,'image_left','*.png')))
                self.depth[seq] = sorted(glob.glob(path.join(seq,'depth_left','*.npy')))
                assert(len(self.image[seq])==len(self.depth[seq])==self.poses[seq].shape[0])
                self.sizes.append(len(self.image[seq]))
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with bz2.BZ2File(catalog_path, 'wb') as f:
                pickle.dump((self.sequences, self.image, self.depth, self.poses, self.sizes), f)
        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = 320, 320, 320, 240
        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # include/exclude seq with regex
        incl_pattern = re.compile(include) if include is not None else None
        excl_pattern = re.compile(exclude) if exclude is not None else None
        final_list = []
        for seq, size in zip(self.sequences, self.sizes):
            if (incl_pattern and incl_pattern.search(seq) is None) or \
                    (excl_pattern and excl_pattern.search(seq) is not None):
                del self.poses[seq], self.image[seq], self.depth[seq]
            else:
                final_list.append((seq, size))
        self.sequences, self.sizes = zip(*final_list) if len(final_list) > 0 else ([], [])

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, ret):
        i, frame = ret
        seq, K = self.sequences[i], self.K
        image = Image.open(self.image[seq][frame])
        depth = F.to_pil_image(np.load(self.depth[seq][frame]), mode='F')
        pose = self.poses[seq][frame]
        image, K, depth = self.augment(image, self.K, depth)
        return image, depth, pose, K, seq.split(os.path.sep)[-3:]

    def rand_split(self, ratio, seed=42):
        total, ratio = len(self.sequences), np.array(ratio)
        split_idx = np.cumsum(np.round(ratio / sum(ratio) * total), dtype=np.int)[:-1]
        subsets = []
        for perm in np.split(np.random.default_rng(seed=seed).permutation(total), split_idx):
            subset = copy(self)
            subset.sequences = np.take(self.sequences, perm).tolist()
            subset.sizes = np.take(self.sizes, perm).tolist()
            subsets.append(subset)
        return subsets


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


class AirSampler(Sampler):
    def __init__(self, data, batch_size, shuffle='all', overlap=True):
        self.seq_sizes = [(seq_id, size) for seq_id, size in enumerate(data.sizes)]
        if shuffle == 'seq': np.random.shuffle(self.seq_sizes)
        self.bs = batch_size
        self.batches = []
        for seq_id, size in self.seq_sizes:
            b_start = np.arange(0, size - self.bs, 1 if overlap else self.bs)
            self.batches += [list(zip([seq_id]*self.bs, range(st, st+self.bs))) for st in b_start]
        if shuffle == 'all': np.random.shuffle(self.batches)

    def __iter__(self):
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
