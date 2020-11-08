#!/usr/bin/env python3

import os
import glob
import torch
import numpy as np
from os import path
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class TartanAir(Dataset):
    def __init__(self, root, scale=1, transform=None, depth_transform=None):
        super().__init__()
        self.transform, self.depth_transform = transform, depth_transform
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
        fx, fy, cx, cy = 320, 320, 320, 240
        K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy]]) * scale
        self.K = torch.cat([K, torch.FloatTensor([0, 0, 1]).view(1,3)], dim=0)

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, ret):
        i, frame = ret
        seq = self.sequences[i]
        image = Image.open(self.image[seq][frame])
        depth = torch.from_numpy(np.load(self.depth[seq][frame])).unsqueeze(0)
        image = image if self.transform is None else self.transform(image)
        depth = depth if self.depth_transform is None else self.depth_transform(depth)
        pose = self.poses[seq][frame]
        return image, depth, pose, self.K


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

    data = TartanAir('/data/datasets/tartanair', scale=1, transform=T.ToTensor())
    sampler = AirSampler(data, batch_size=4, shuffle=True)
    loader = DataLoader(dataset=data,  batch_sampler=sampler, num_workers=4, pin_memory=True)

    for i, (image, depth, pose, K) in enumerate(loader):
        print(i, image.shape, depth.shape, pose.shape, K.shape)
