#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Visualization
from utils import PairwiseProjector
import kornia.geometry.conversions as C
from models.featurenet import GridSample


class FeatureNetLoss(nn.Module):
    def __init__(self, alpha=[1, 1, 1], K=None, debug=False):
        super().__init__()
        self.alpha = alpha
        self.sample = GridSample()
        self.distinction = DistinctionLoss()
        self.projection = ScoreProjectionLoss()
        self.projector = PairwiseProjector(K)
        self.match = DiscriptorMatchLoss(debug=debug)
        self.debug = Visualization('loss') if debug else debug

    def forward(self, features, points, scores_dense, depths_dense, poses, K, imgs):
        scores = self.sample((scores_dense, points))
        distinction = self.distinction(features, scores)
        proj_pts, invis_idx = self.projector(points, depths_dense, poses, K)
        projection = self.projection(scores_dense, scores, proj_pts, invis_idx)
        match = self.match(features, points, proj_pts, invis_idx, *scores_dense.shape[2:4])

        if self.debug is not False:
            src_idx, dst_idx, pts_idx = invis_idx
            _proj_pts = proj_pts.clone()
            _proj_pts[src_idx, dst_idx, pts_idx, :] = -2
            for dbgpts in _proj_pts:
                self.debug.show(imgs, dbgpts)

        return self.alpha[0] * distinction + self.alpha[1] * projection + self.alpha[2] * match


class DistinctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.bceloss = nn.BCELoss()

    def forward(self, features, scores):
        features = F.normalize(features, dim=1).detach()
        summation = features.sum(dim=1, keepdim=True).transpose(1, 2)
        similarity = (features@summation - 1)/(features.size(1) - 1)
        targets = 1 - self.relu(similarity)
        return self.bceloss(scores, targets)


class ScoreProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample = GridSample()
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, scores_dense, scores_src, proj_pts, invis_idx):
        scores_dst = self.sample((scores_dense, proj_pts))
        scores_src = scores_src.unsqueeze(1).expand_as(scores_dst)
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
        pts_src = pts_src.unsqueeze(1).expand_as(pts_dst).reshape(B**2, N, 2)
        pts_dst = pts_dst.reshape_as(pts_src)

        idx = (torch.cdist(pts_src, pts_dst)<=self.radius).nonzero().T
        src, dst = [idx[0]%B, idx[1]], [idx[0]//B, idx[2]]
        cosine = self.cosine(features[src], features[dst])

        if self.debug:
            for b, n, b1, n1 in zip(src[0], src[1], dst[0], dst[1]):
                print("%d <-> %d, %d <-> %d (2)" % (b, b1, n, n1))

        return (1 - cosine).mean()
