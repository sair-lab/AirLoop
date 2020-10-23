#!/usr/bin/env python3

import glob
import torch
import numpy as np
from os import path
from PIL import Image
from torchvision.datasets import VisionDataset

from utils import pose2mat


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
            return image, depth, pose
        else:
            return image


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
