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
    Better using torchvision.ops.nms
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


class GridSample(nn.Module):
    '''
    '''
    def __init__(self, scale_factor, mode='bilinear'):
        super().__init__()
        self.scale_factor, self.mode = scale_factor, mode

    def forward(self, inputs):
        features, points = inputs
        size = torch.Tensor([features.shape[2:4]]).to(features)
        points = [2*p/(size*self.scale_factor-1)-1 for p in points]
        output = [F.grid_sample(features[i:i+1], p.view(1,1,-1,2), 
                    self.mode, align_corners=True) for i,p in enumerate(points)]
        return torch.cat(output, dim=-1).squeeze().t()


class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, alpha, dropout=0.5, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1), nn.Dropout(dropout))
        self.leakyrelu = nn.LeakyReLU(beta)

    def forward(self, x):
        h = self.tran(x)
        att = self.att1(h).unsqueeze(0) + self.att2(h).unsqueeze(1)
        adj = self.norm(self.leakyrelu(att.squeeze()))
        return self.alpha * h + (1-self.alpha) * adj @ h


class FeatureNet(models.VGG):

    feat_dim = 512
    radius = 4
    score_threshold = 0.005
    max_keypoints = -1
    zero_border = 4

    def __init__(self):
        super().__init__(models.vgg13().features)
        # Only take the first 19 layers of pre-trained vgg13. Feature Map: (512, H/8, W/8)
        self.load_state_dict(models.vgg13(pretrained=True).state_dict())
        self.features = nn.Sequential(*list(self.features.children())[:19])
        del self.classifier

        self.scores = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(512, 65, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1),
                IndexSelect(dim=1, index=torch.LongTensor(list(range(64)))),
                nn.PixelShuffle(upscale_factor=8),
                NMS(radius=4, iteration=2),
                ZeroBorder(self.zero_border))

        self.descriptors = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(512, self.feat_dim, kernel_size=1, stride=1, padding=0))
        self.sample = nn.Sequential(GridSample(scale_factor=8), nn.BatchNorm1d(self.feat_dim))
        self.encoder = nn.Sequential(nn.Linear(3,256),nn.ReLU(),nn.Linear(256,self.feat_dim))

        self.graph = nn.Sequential(
                GraphAttn(in_features=512, out_features=256, alpha=0.9), nn.ReLU(),
                GraphAttn(in_features=256, out_features=256, alpha=0.9))

    def forward(self, inputs):

        features = self.features(inputs)

        scores = self.scores(features)

        b, c, h, w = (scores > self.score_threshold).nonzero(as_tuple=True)

        nums = [(b==i).sum() for i in range(inputs.size(0))]

        scores = scores[b,c,h,w].view(-1,1)

        points = torch.stack((h,w), dim=1)

        descriptors = self.descriptors(features)

        descriptors = self.sample((descriptors, points.split(nums)))

        nodes = descriptors + self.encoder(torch.cat([points, scores],dim=-1))

        features = [self.graph(n) for n in nodes.split(nums)]

        return points.split(nums), scores.split(nums), features


if __name__ == "__main__":
    '''Test codes'''
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Test FeatureNet')
    parser.add_argument("--device", type=str, default='cuda', help="cuda, cuda:0, or cpu")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("--batch-size", type=int, default=30, help="number of minibatch size")
    parser.add_argument('--crop-size', nargs='+', type=int, default=[320,320], help='image crop size')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = FeatureNet().to(args.device)
    inputs = torch.randn(args.batch_size,3,*args.crop_size).to(args.device)

    start = time.time()
    with torch.no_grad():
        for i in range(5):
            points, scores, descriptors = net(inputs)
            torch.cuda.empty_cache()
            for i in range(len(points)):
                print(i, 'P:',points[i].shape, 'S:',scores[i].shape, 'D:',descriptors[i].shape)
    print('time:', time.time()-start)
