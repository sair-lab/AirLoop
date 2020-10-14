#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class IndexSelect(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim, self.index = dim, index

    def forward(self, x):
        self.index = self.index.to(x.device)
        return x.index_select(self.dim, self.index)


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1, eps=1e-12):
        super().__init__()
        self.p, self.dim, self.eps = p, dim, eps

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, self.eps)


class NMS(nn.Module):
    '''
    Non-Maximum Suppression (Temporary Implementation)
    Adopted from SupperGlue Implementation
    Better implementation using torchvision.ops.nms
    '''
    def __init__(self, radius=4, iteration=2):
        super().__init__()
        assert(radius >= 0)
        self.iteration = iteration
        self.pool = nn.MaxPool2d(kernel_size=radius*2+1, stride=1,padding=radius)

    def forward(self, scores):
        zeros = torch.zeros_like(scores)
        max_mask = scores == self.pool(scores)
        for _ in range(self.iteration):
            supp_mask = self.pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == self.pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)


class ZeroBorder(nn.Module):
    '''
    Set Boarders to Zero
    '''
    def __init__(self, border=4):
        super().__init__()
        self.pad1 = nn.ZeroPad2d(-border)
        self.pad2 = nn.ZeroPad2d(border)

    def forward(self, x):
        return self.pad2(self.pad1(x))


class FeatureNet(models.VGG):

    feat_dim = 512
    radius = 4
    score_threshold = 0.005
    max_keypoints = -1
    zero_border = 4

    def __init__(self):
        super().__init__(models.vgg13().features)
        # Only take first 19 layers of pre-trained vgg13
        # Output dimension: (512, H/8, W/8)
        self.load_state_dict(models.vgg13(pretrained=True).state_dict())
        del self.classifier
        self.features = nn.Sequential(*list(self.features.children())[:19])

        # Compute scores for each input pixel
        self.scores = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(512, 65, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1),
                IndexSelect(dim=1, index=torch.LongTensor(list(range(64)))),
                nn.PixelShuffle(upscale_factor=8),
                NMS(radius=4, iteration=2),
                ZeroBorder(self.zero_border))

        self.descriptors = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(512, self.feat_dim, kernel_size=1, stride=1, padding=0),
                Normalize(p=2, dim=1))


    def forward(self, inputs):

        features = self.features(inputs)

        full_scores = self.scores(features)

        keypoints = (full_scores > self.score_threshold).nonzero(as_tuple=False)

        scores = full_scores[tuple(keypoints.t())]

        # keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        descriptors = self.descriptors(features)

        return keypoints, scores, descriptors


if __name__ == "__main__":
    '''Test codes'''
    import os, argparse
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    from torchvision.datasets import CocoDetection

    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument("--device", type=str, default='cuda', help="cuda:0 or cpu")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("--batch-size", type=int, default=2, help="number of minibatch size")
    parser.add_argument('--crop-size', nargs='+', type=int, default=[320,320], help='image crop size')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = FeatureNet().to(args.device)

    with torch.no_grad():
        inputs = torch.randn(args.batch_size,3,*args.crop_size).to(args.device)
        outputs = net(inputs)
