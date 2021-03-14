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
    def __init__(self, in_features, out_features, alpha=0.9, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.tran = nn.Linear(in_features, out_features)
        self.att1 = nn.Linear(out_features, 1)
        self.att2 = nn.Linear(out_features, 1)
        self.actv = nn.Sequential(nn.LeakyReLU(beta), nn.Softmax(dim=-1))

    def forward(self, x):
        h = self.tran(x)
        att = self.att1(h) + self.att2(h).permute(0, 2, 1)
        adj = self.actv(att.squeeze())
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
    def __init__(self, feat_dim, feat_num, sample_pass):
        super().__init__()
        self.feat_dim, self.feat_num, self.sample_pass = feat_dim, feat_num, sample_pass

        self.descriptor = nn.Sequential(
            make_layer(256, self.feat_dim),
            make_layer(self.feat_dim, self.feat_dim, bn=None, activation=None))
        self.sample = nn.Sequential(GridSample(), nn.BatchNorm1d(self.feat_num))
        self.residual = nn.Sequential(make_layer(3, 128, kernel_size=9, padding=4), make_layer(128, self.feat_dim))

    def forward(self, images, features, points, scores):
        descriptors, residual = self.descriptor(features), self.residual(images)
        n_group = 1 + self.sample_pass if self.training else 1
        descriptors, residual = _repeat_flatten(descriptors, n_group), _repeat_flatten(residual, n_group)
        descriptors = self.sample((descriptors, points)) + self.sample((residual, points))
        return descriptors


class GDNet(nn.Module):
    def __init__(self, feat_dim, desc_dim, n_heads=4, n_pass=1):
        super().__init__()
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(feat_dim, n_heads)
            for _ in range(n_pass)])
        self.weight = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, 1), nn.LeakyReLU(),
        )
        self.content = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim),
        )

    def forward(self, features):
        for at in self.attn:
            features = self._res_attend(at, features, features)
        weights, content = self.weight(features), self.content(features)
        return (weights.transpose(1, 2) @ content).squeeze(1)

    def _res_attend(self, attn, q_desc, kv_desc):
        # nn.MultiheadAttention uses axis order (N, B, D)
        q, kv = q_desc.permute(1, 0, 2), kv_desc.permute(1, 0, 2)
        return attn.forward(q, kv, kv, need_weights=False)[0].permute(1, 0, 2)


class FeatureNet(models.VGG):
    def __init__(self, feat_dim=256, feat_num=500, sample_pass=1):
        super().__init__(models.vgg13().features)
        self.feat_dim, self.feat_num, self.sample_pass = feat_dim, feat_num, sample_pass
        # Only adopt the first 15 layers of pre-trained vgg13. Feature Map: (512, H/8, W/8)
        self.load_state_dict(models.vgg13(pretrained=True).state_dict())
        self.features = nn.Sequential(*list(self.features.children())[:15])
        del self.classifier

        self.scores = ScoreHead(8)
        self.descriptors = DescriptorHead(feat_dim, feat_num, sample_pass)
        self.global_desc = GDNet(feat_dim, feat_dim)
        self.nms = nms.NonMaximaSuppression2d((5, 5))

    def forward(self, inputs):

        B, _, H, W = inputs.shape

        features = self.features(inputs)

        pointness = self.scores(inputs, features)

        scores, points = self.nms(pointness).view(B, -1, 1).topk(self.feat_num, dim=1)

        points = torch.cat((points % W, points // W), dim=-1)

        n_group = 1
        if self.training:
            n_group += self.sample_pass
            scores_flat_dup = _repeat_flatten(pointness.view(B, H * W), self.sample_pass)
            points_rand = torch.multinomial(torch.ones_like(scores_flat_dup), self.feat_num)
            scores_rand = torch.gather(scores_flat_dup, 1, points_rand).unsqueeze(-1)
            points_rand = torch.stack((points_rand % W, points_rand // W), dim=-1)
            points = self._append_group(points_rand, self.sample_pass, points).reshape(B * n_group, self.feat_num, 2)
            scores = self._append_group(scores_rand, self.sample_pass, scores).reshape(B * n_group, self.feat_num, 1)

        points = C.normalize_pixel_coordinates(points, H, W)

        descriptors = self.descriptors(inputs, features, points, scores)

        gd = self.global_desc(descriptors[((n_group - 1) * B):])

        N = n_group * self.feat_num
        return descriptors.reshape(B, N, self.feat_dim), points.reshape(B, N, 2), pointness, scores.reshape(B, N), gd

    @staticmethod
    def _append_group(grouped_samples, sample_pass, new_group):
        """(B*S, N, *) + (B, N, *) -> (B*(S+1), N, *)"""
        BS, *_shape = grouped_samples.shape
        raveled = grouped_samples.reshape(BS // sample_pass, sample_pass, *_shape)
        return torch.cat((raveled, new_group.unsqueeze(1)), dim=1)


def make_layer(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bn=nn.BatchNorm2d, activation=nn.ReLU()):
    modules = [nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding)] + \
        ([bn(out_chan)] if bn is not None else []) + \
        ([activation] if activation is not None else [])
    return nn.Sequential(*modules)


def _repeat_flatten(x, n):
    """[B0, B1, B2, ...] -> [B0, B0, ..., B1, B1, ..., B2, B2, ...]"""
    shape = x.shape
    return x.unsqueeze(1).expand(shape[0], n, *shape[1:]).reshape(shape[0] * n, *shape[1:])


if __name__ == "__main__":
    '''Test codes'''
    import argparse
    from .tool import Timer

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
