#!/usr/bin/env python3

import os
import glob
import torch
import numpy as np
from os import path
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from scipy.spatial.transform import Rotation as R


class DatasetBase(VisionDataset):
    def __init__(self, root='/data/',
                 img_dir='.', img_ext=['jpg', 'png'], img_transform=None,
                 depth_dir=None, depth_ext=['npy'], depth_transform=None,
                 pose_file=None,
                 train=True, download=None):
        super(DatasetBase, self).__init__(root)

        self.img_files = self._dump_dir(img_dir, img_ext)
        self.size = len(self.img_files)

        if depth_dir:
            self.depth_files = self._dump_dir(depth_dir, depth_ext)
            assert(len(self.depth_files) == self.size)
        else:
            self.depth_files = None

        pose_file_full = path.join(self.root, pose_file)
        if pose_file is not None and path.isfile(pose_file_full):
            self.poses = self.load_poses(pose_file_full)
            assert(self.poses.shape == (self.size, 3, 4))
        else:
            self.poses = None

        self.train = train
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        
        # TEMP
        fx = 320.0 / 2
        fy = 320.0 / 2
        cx = 320.0 / 2
        cy = 240.0 / 2
        self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # TEMP

    def _dump_dir(self, directory, exts, sort=True):
        files = []
        for ext in exts:
            files.extend(
                glob.glob(path.join(self.root, directory, '*.%s' % ext)))
        return sorted(files) if sort else files

    def load_img(self, img_file):
        return Image.open(img_file)

    def load_depth(self, depth_file):
        return torch.from_numpy(np.load(depth_file))

    def load_poses(self, pose_file):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = self.load_img(self.img_files[index])
        image = self.img_transform(image) if self.img_transform else image
        if self.train is True:
            depth = self.load_depth(self.depth_files[index])
            depth = self.depth_transform(depth) if self.depth_transform else depth
            pose = self.poses[index]
            return image, depth, pose, self.K
        else:
            return image, self.K


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
        ned2den = torch.tensor([[0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0]]).to(dtype=torch.float32)
        return ned2den @ pose2mat(poses7)


class tartanair(Dataset):
    def __init__(self, root='/data/datasets/tartanair', scale=0.5, transform=None, depth_transform=None):
        super().__init__()
        self.transform, self.depth_transform = transform, depth_transform
        self.sequences = glob.glob(os.path.join(root,'*','Easy','*'))
        self.image, self.depth, self.poses, self.size = {}, {}, {}, torch.IntTensor()
        ned2den = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        for seq in self.sequences:
            quaternion = np.loadtxt(path.join(seq, 'pose_left.txt'), dtype=np.float32)
            self.poses[seq] = ned2den @ pose2mat(quaternion)
            self.image[seq] = sorted(glob.glob(path.join(seq,'image_left','*.png')))
            self.depth[seq] = sorted(glob.glob(path.join(seq,'depth_left','*.npy')))
            assert(len(self.image[seq])==len(self.depth[seq])==self.poses[seq].shape[0])
            self.size = torch.cat([self.size, torch.IntTensor([len(self.image[seq])])])
        self.sizecum = self.size.cumsum(dim=0)

    def __len__(self):
        return self.size.sum()

    def __getitem__(self, frame):
        i = 0
        while self.size[i] - frame <= 0:
            frame = frame - self.size[i]
            i = i + 1
        # return i, frame
        seq = self.sequences[i]
        image = Image.open(self.image[seq][frame])
        depth = torch.from_numpy(np.load(self.depth[seq][frame]))
        pose = self.poses[seq][frame]
        image = image if self.transform is None else self.transform(image)
        return image, depth, pose



class AirSampler(Sampler):

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.batch_size = data, batch_size
        self.shuffle = shuffle
        self.len = len(data) - batch_size
        self.size = data.size

    def __iter__(self):
        L = torch.randperm(self.len) if self.shuffle else torch.arange(self.len)
        self.minibatches = [range(L[n], L[n]+self.batch_size) for n in range(len(L))]
        return iter(self.minibatches)

    def __len__(self):
        return len(self.minibatches)

    def __repr__(self):
        return 'AirSampler(batch_size={})'.format(self.batch_size)




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
    from torchvision import transforms as T
    data = tartanair(transform=T.ToTensor(), depth_transform=T.ToTensor())
    sampler = AirSampler(data, batch_size=5, shuffle=False)
    loader = DataLoader(dataset=data,  batch_sampler=sampler, num_workers=0)

    for i, data in enumerate(loader):
        print(data)
