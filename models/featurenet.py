#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
from kornia.feature import nms
from torchvision import models
import torch.nn.functional as F
import kornia.geometry.conversions as C


class IndexSelect(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim, self.index = dim, index

    def forward(self, x):
        self.index = self.index.to(x.device)
        return x.index_select(self.dim, self.index)


class ConstantBorder(nn.Module):
    '''
    Set Boarders to Constant
    '''
    def __init__(self, border=4, value=-math.inf):
        super().__init__()
        self.pad1 = nn.ConstantPad2d(-border, value=value)
        self.pad2 = nn.ConstantPad2d(border, value=value)

    def forward(self, x):
        return self.pad2(self.pad1(x))


class GridSample(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, inputs):
        features, points = inputs
        dim = len(points.shape)
        points = points.view(features.size(0), 1, -1, 2) if dim == 3 else points
        output = F.grid_sample(features, points, self.mode, align_corners=True).permute(0, 2, 3, 1)
        return output.squeeze(1) if dim == 3 else output


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.factor = downscale_factor

    def forward(self, x):
        (N, C, H, W), S = x.shape, self.factor
        H, W = H // S, W // S
        x = x.reshape(N, C, H, S, W, S).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(N, C * S**2, H, W)
        return x


class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, alpha, dropout=0.5, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.leakyrelu = nn.LeakyReLU(beta)

    def forward(self, x):
        h = self.tran(x)
        att = self.att1(h) + self.att2(h).permute(0, 2, 1)
        adj = self.norm(self.leakyrelu(att.squeeze()))
        return self.alpha * h + (1-self.alpha) * adj @ h


class BatchNorm2dwC(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.bn = nn.BatchNorm3d(1)

    def forward(self, x):
        return self.bn(x.unsqueeze(1)).squeeze(1)


class ScoreHead(nn.Module):
    def __init__(self, in_scale):
        super().__init__()
        self.scores_vgg = nn.Sequential(make_layer(256, 128), make_layer(128, 64, bn=BatchNorm2dwC))
        self.scores_img = nn.Sequential(make_layer(3, 8), make_layer(8, 16, bn=BatchNorm2dwC),
            PixelUnshuffle(downscale_factor=in_scale))
        self.combine = nn.Sequential(
            make_layer(64 + 16 * in_scale**2, in_scale**2 + 1, bn=BatchNorm2dwC, activation=nn.Softmax(dim=1)),
            IndexSelect(dim=1, index=torch.arange(in_scale**2)),
            nn.PixelShuffle(upscale_factor=in_scale),
            ConstantBorder(border=4, value=0))

    def forward(self, images, features):
        scores_vgg, scores_img = self.scores_vgg(features), self.scores_img(images)
        return self.combine(torch.cat([scores_vgg, scores_img], dim=1))


class DescriptorHead(nn.Module):
    def __init__(self, feat_dim, feat_num):
        super().__init__()
        self.feat_dim, self.feat_num = feat_dim, feat_num

        self.descriptor = nn.Sequential(
            make_layer(256, self.feat_dim),
            make_layer(self.feat_dim, self.feat_dim, bn=None, activation=None))
        self.sample = nn.Sequential(GridSample(), nn.BatchNorm1d(self.feat_num))
        self.encoder = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, self.feat_dim))
        self.residual = nn.Sequential(make_layer(3, 128, kernel_size=9, padding=4), make_layer(128, self.feat_dim))

    def forward(self, images, features, points, scores):
        descriptors, residual = self.descriptor(features), self.residual(images)
        descriptors = self.sample((descriptors, points)) + self.sample((residual, points))
        descriptors = descriptors + self.encoder(torch.cat([points, scores], dim=-1))
        return descriptors


class FeatureNet(models.VGG):
    def __init__(self, feat_dim=256, feat_num=500):
        super().__init__(models.vgg13().features)
        self.feat_dim, self.feat_num = feat_dim, feat_num
        # Only adopt the first 15 layers of pre-trained vgg13. Feature Map: (512, H/8, W/8)
        self.load_state_dict(models.vgg13(pretrained=True).state_dict())
        self.features = nn.Sequential(*list(self.features.children())[:15])
        del self.classifier

        self.scores = ScoreHead(8)
        self.descriptors = DescriptorHead(feat_dim, feat_num)
        self.graph = nn.Sequential(
            GraphAttn(self.feat_dim, self.feat_dim, alpha=0.9), nn.ReLU(),
            GraphAttn(self.feat_dim, self.feat_dim, alpha=0.9))
        self.nms = nms.NonMaximaSuppression2d((5, 5))

    def forward(self, inputs):

        B, _, H, W = inputs.shape

        features = self.features(inputs)

        pointness = self.scores(inputs, features)

        scores, points = self.nms(pointness).view(B, -1, 1).topk(self.feat_num, dim=1)

        points = torch.cat((points % W, points // W), dim=-1)

        points = C.normalize_pixel_coordinates(points, H, W)

        descriptors = self.descriptors(inputs, features, points, scores)

        descriptors = self.graph(descriptors)

        return descriptors, points, pointness, scores


def make_layer(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bn=nn.BatchNorm2d, activation=nn.ReLU()):
    modules = [nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding)] + \
        ([bn(out_chan)] if bn is not None else []) + \
        ([activation] if activation is not None else [])
    return nn.Sequential(*modules)


if __name__ == "__main__":
    '''Test codes'''
    import argparse
    from tool import Timer

    parser = argparse.ArgumentParser(description='Test FeatureNet')
    parser.add_argument("--device", type=str, default='cuda', help="cuda, cuda:0, or cpu")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument('--crop-size', nargs='+', type=int, default=[320, 320], help='image crop size')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = FeatureNet(512, 200).to(args.device).eval()
    inputs = torch.randn(args.batch_size, 3, *args.crop_size).to(args.device)

    timer = Timer()
    with torch.no_grad():
        for i in range(5):
            descriptors, points, pointness, scores = net(inputs)
            print('%d D: %s, P: (%s, %s), S: %s' % (i, descriptors.shape, pointness.shape, points.shape, scores.shape))
    print('time:', timer.end())
