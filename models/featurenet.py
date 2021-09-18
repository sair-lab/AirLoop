#!/usr/bin/env python3

from functools import partial

import numpy as np

import torch
import torch.nn as nn
from kornia.feature import nms
from torchvision import models
import torch.nn.functional as F
import kornia.geometry.conversions as C

from utils import GridSample


class GeM(nn.Module):
    def __init__(self, feat_dim, desc_dim, p=3):
        super().__init__()
        self.p = p
        self.whiten = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.LeakyReLU(),
            nn.Linear(desc_dim, desc_dim)
        )

    def forward(self, features):
        mean = (features ** self.p).mean(dim=1)
        return self.whiten(mean.sign() * mean.abs() ** (1 / self.p)), None


class FeatureNet(nn.Module):
    def __init__(self, res=(240, 320), feat_dim=256, feat_num=500, gd_dim=1024, sample_pass=0, gd_only=False):
        super().__init__()

        vgg = models.vgg19(pretrained=True)
        vgg.avgpool = vgg.classifier = nn.Identity()
        self.features = vgg

        self.gd_indim = 512
        self.global_desc = GeM(self.gd_indim, gd_dim)
        self.gd_only = gd_only

    def forward(self, img):

        B, _, H, W = img.shape

        fea = self.features(img)

        gd, gd_locs = self.global_desc(fea.reshape(B, self.gd_indim, fea.shape[-1] * fea.shape[-2]).transpose(-1, -2))

        return (gd, gd_locs) if self.training else gd
