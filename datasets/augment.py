#!/usr/bin/env python3

import torch
import numpy as np
import kornia as kn
from torch import nn
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class AirAugment(nn.Module):
    def __init__(self, scale=1, size=[480, 640], resize_only=False):
        super().__init__()
        self.img_size = (np.array(size) * scale).round().astype(np.int32)
        self.resize_totensor = T.Compose([T.Resize(self.img_size.tolist()), np.array, T.ToTensor()])
        self.rand_crop = T.RandomResizedCrop(self.img_size.tolist(), scale=(0.1, 1.0))
        self.rand_rotate = T.RandomRotation(45, resample=Image.BILINEAR)
        self.rand_color = T.ColorJitter(0.8, 0.8, 0.8)
        self.p = [1, 0, 0, 0] if resize_only else [0.25]*4

    def apply_affine(self, K, translation=[0, 0], center=[0, 0], scale=[1, 1], angle=0):
        """Applies transformation to K in the order: (R, S), T. All coordinates are in (h, w) order.
           Center is for both scale and rotation.
        """
        translation = torch.tensor(translation[::-1].copy(), dtype=torch.float32)
        center = torch.tensor(center[::-1].copy(), dtype=torch.float32)
        scale = torch.tensor(scale[::-1].copy(), dtype=torch.float32)
        angle = torch.tensor([angle], dtype=torch.float32)

        scaled_rotation = torch.block_diag(kn.angle_to_rotation_matrix(angle)[0] @ torch.diag(scale), torch.ones(1))
        scaled_rotation[:2, 2] = center - scaled_rotation[:2, :2] @ center + translation

        return scaled_rotation.to(K) @ K

    def forward(self, image, K, depth=None):
        if isinstance(image, Image.Image):
            image = self.resize_totensor(image)
            depth = depth if depth is None else self.resize_totensor(depth)
        elif isinstance(image, torch.Tensor):
            image = self.resize_totensor.transforms[0](image)
            depth = depth if depth is None else self.resize_totensor.transforms[0](depth)

        in_size = np.array(image.shape[1:])
        center, scale, angle = in_size/2, self.img_size/in_size, 0

        transform = np.random.choice(np.arange(len(self.p)), p=self.p)
        if transform == 1:
            trans = self.rand_crop
            i, j, h, w = T.RandomResizedCrop.get_params(image, trans.scale, trans.ratio)
            center = np.array([i + h / 2, j + w / 2])
            scale = self.img_size / np.array([h, w])
            image = F.resized_crop(image, i, j, h, w, trans.size, trans.interpolation)
            depth = depth if depth is None else F.resized_crop(depth, i, j, h, w, trans.size, trans.interpolation)

        elif transform == 2:
            trans = self.rand_rotate
            angle = T.RandomRotation.get_params(trans.degrees)
            # fill oob pix with reflection so that model can't detect rotation with boundary
            image = F.pad(image, padding=tuple(in_size // 2), padding_mode='reflect')
            image = F.rotate(image, angle, trans.resample, trans.expand, trans.center, trans.fill)
            image = F.center_crop(image, tuple(in_size))
            # fill oob depth with inf so that projector can mask them out
            if depth is not None and isinstance(depth, torch.Tensor):
                # torch 1.7.1: F.rotate doesn't support fill for Tensor
                device = depth.device
                depth = F.to_pil_image(depth, mode='F')
                depth = F.rotate(depth, angle, trans.resample, trans.expand, trans.center, float('inf'))
                depth = self.resize_totensor(depth).to(device)

        elif transform == 3:
            image = self.rand_color(image)

        translation = self.img_size / 2 - center
        K = self.apply_affine(K, translation, center, scale, angle)

        return (image, K) if depth is None else (image, K, depth)
