#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Visualization
from utils import PairwiseProjector
from models.featurenet import GridSample
from kornia import PinholeCamera, DepthWarper


class FeatureNetLoss(nn.Module):
    def __init__(self, height, width, alpha=[1,1,1], K=None, debug=True):
        super().__init__()
        self.alpha = alpha
        self.sample = GridSample()
        self.distinction = DistinctionLoss()
        self.projection = ScoreProjectionLoss()
        self.projector = PairwiseProjector(width, height, K)
        self.debug = Visualization('loss') if debug else debug

    def forward(self, features, points, scores_dense, depths_dense, poses, K, imgs):
        scores = self.sample((scores_dense, points))
        proj_pts, invis_idx = self.projector(points, depths_dense, poses, K)
        distinction = self.distinction(features, scores)
        projection = self.projection(scores_dense, scores, proj_pts.transpose(0, 1), invis_idx)

        if self.debug is not False:
            src_idx, dst_idx, pts_idx = invis_idx
            _proj_pts = proj_pts.clone()
            _proj_pts[src_idx, dst_idx, pts_idx, :] = -2
            for dbgpts in _proj_pts:
                self.debug.show(imgs, dbgpts)

        return self.alpha[0]*distinction + self.alpha[1]*projection


class DistinctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()
        self.relu = nn.ReLU()

    def forward(self, features, scores):
        features = F.normalize(features, dim=1).detach()
        summation = features.sum(dim=1, keepdim=True).transpose(1, 2)
        similarity = (features@summation - 1)/(features.size(1) - 1)
        targets = 1 - self.relu(similarity)
        return self.bceloss(scores, targets)


class ScoreProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction='none')
        self.sample = GridSample()

    def forward(self, scores_dense, scores_src, proj_pts, invis_idx):
        scores_dst = self.sample((scores_dense, proj_pts))
        scores_src_dup = scores_src.unsqueeze(0).expand_as(scores_dst)
        score_proj_loss = self.mseloss(scores_dst, scores_src_dup)
        src_idx, dst_idx, pts_idx = invis_idx
        score_proj_loss[src_idx, dst_idx, pts_idx] = 0
        score_proj_loss = score_proj_loss.mean()
        return score_proj_loss


def reproj_error(p, d_p, T_p, T_q, q, K, K_inv=None, red='none'):
    """MSE of reprojection.

    Args:
      p, q:     Pixel coordinates (N, 2).
      T_p, T_q: Poses where p, q are observed (N, 3, 4).
      K, K_inv: Camera intrinsics (N, 3, 3).
      red:      Optional. Reduce to one number?

    Returns:
      MSE(reproject(p), q).
    """
    return F.mse_loss(project_points(p, d_p, T_p, T_q, K, K_inv), q, reduction=red).sum(1)
