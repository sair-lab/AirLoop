#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Visualization
from utils import PairwiseProjector
import kornia.geometry.conversions as C
from models.featurenet import GridSample


class FeatureNetLoss(nn.Module):
    def __init__(self, beta=[1, 1, 1], K=None, debug=False):
        super().__init__()
        self.beta = beta
        self.sample = GridSample()
        self.distinction = DistinctionLoss()
        self.projection = ScoreProjectionLoss()
        self.projector = PairwiseProjector(K)
        self.match = DiscriptorMatchLoss(debug=debug)
        self.debug = Visualization('loss') if debug else debug

    def forward(self, features, points, pointness, depths_dense, poses, K, imgs):
        H, W = pointness.size(2), pointness.size(3)
        scores = self.sample((pointness, points))
        distinction = self.distinction(features, scores)
        proj_pts, invis_idx = self.projector(points, depths_dense, poses, K)
        projection = self.projection(pointness, scores, proj_pts, invis_idx)
        match = self.match(features, points, proj_pts, invis_idx, H, W)

        if self.debug is not False:
            print('Loss', distinction, projection, match)
            src_idx, dst_idx, pts_idx = invis_idx
            _proj_pts = proj_pts.clone()
            _proj_pts[src_idx, dst_idx, pts_idx, :] = -2
            for dbgpts in _proj_pts:
                self.debug.show(imgs, dbgpts)
            self.debug.showmatch(imgs[0], points[0], imgs[1], proj_pts[0,1])

        return self.beta[0]*distinction + self.beta[1]*projection + self.beta[2]*match


class DistinctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.bceloss = nn.BCELoss()
        self.pcosine = PairwiseCosineSimilarity()

    def forward(self, features, scores):
        features = F.normalize(features, dim=2).detach()
        summation = features.sum(dim=1, keepdim=True).transpose(1, 2)
        similarity = (features@summation - 1)/(features.size(1) - 1)
        targets = 1 - self.relu(similarity)
        feat_loss = 1 - self.pcosine(features, features).mean()
        return self.bceloss(scores, targets) + feat_loss


class ScoreProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample = GridSample()
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, pointness, scores_src, proj_pts, invis_idx):
        scores_dst = self.sample((pointness, proj_pts))
        scores_src = scores_src.unsqueeze(0).expand_as(scores_dst)
        proj_loss = self.mseloss(scores_dst, scores_src)
        src_idx, dst_idx, pts_idx = invis_idx
        proj_loss[src_idx, dst_idx, pts_idx] = 0
        return proj_loss.mean()


class DiscriptorMatchLoss(nn.Module):
    def __init__(self, radius=1, debug=False):
        super(DiscriptorMatchLoss, self).__init__()
        self.radius, self.debug = radius, debug
        self.cosine = nn.CosineSimilarity()

    def forward(self, features, pts_src, pts_dst, invis_idx, height, width):
        B, N, _ = pts_src.shape

        pts_src = C.denormalize_pixel_coordinates(pts_src.detach(), height, width)
        pts_dst = C.denormalize_pixel_coordinates(pts_dst.detach(), height, width)
        pts_src = pts_src.unsqueeze(0).expand_as(pts_dst).reshape(B**2, N, 2)
        pts_dst = pts_dst.reshape_as(pts_src)

        match = torch.cdist(pts_src, pts_dst)<=self.radius
        idx = match.triu(diagonal=1).nonzero(as_tuple=True)
        src, dst = [idx[0]%B, idx[1]], [idx[0]//B, idx[2]]
        cosine = self.cosine(features[src], features[dst])

        if self.debug:
            for b, n, b1, n1 in zip(src[0], src[1], dst[0], dst[1]):
                print("%d <-> %d, %d <-> %d (2)" % (b, b1, n, n1))

        return (1 - cosine).mean()


class PairwiseCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=-2)

    def forward(self, x, y):
        x = x.permute((1, 2, 0))
        y = y.permute((1, 2, 0)).unsqueeze(1)
        c = self.cosine(x, y)
        return c.permute((2, 0, 1))
