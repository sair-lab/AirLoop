#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia import PinholeCamera, DepthWarper, denormalize_pixel_coordinates

from utils import PairwiseProjector

# TEMP
import cv2
from utils import Visualization
vis = Visualization('src->tgt')
# TEMP


class FeatureNetLoss(nn.Module):
    def __init__(self, height, width, lamb_dist=1, lamb_sproj=1, K=None):
        super().__init__()

        self.distinct_loss = DistinctivenessLoss()
        self.score_proj_loss = ScoreProjectionLoss()

        self.lamb_dist = lamb_dist
        self.lamb_sproj = lamb_sproj

        self.projector = PairwiseProjector(width, height, K)

    def forward(self, features, points, scores_dense, depths_dense, poses, K, imgs):
        scores = F.grid_sample(scores_dense, points[:, None], align_corners=False).squeeze()

        proj_pts, invis_idx = self.projector(points, depths_dense, poses, K)
        # TEMP
        # src_idx, dst_idx, pts_idx = invis_idx
        # _proj_pts = proj_pts.clone()
        # _proj_pts[src_idx, dst_idx, pts_idx, :] = -2
        # for dbgpts in _proj_pts:
        #     vis.show(imgs, dbgpts)
        #     while cv2.waitKey() != ord('n'):
        #         pass
        # TEMP

        score_dist_loss = self.distinct_loss(features, scores)
        score_proj_loss = self.score_proj_loss(scores_dense, scores, proj_pts, invis_idx)

        total_loss = self.lamb_dist * score_dist_loss + self.lamb_sproj * score_proj_loss

        return total_loss


class DistinctivenessLoss(nn.Module):
    def __init__(self):
        super(DistinctivenessLoss, self).__init__()
        self.bceloss = nn.BCELoss()
        self.relu = nn.ReLU()

    def forward(self, features, scores):
        features = F.normalize(features, dim=-1).detach()
        summation = features.sum(dim=1, keepdim=True).transpose(1, 2)
        similarity = (features@summation - 1)/(features.size(1) - 1)
        targets = 1 - self.relu(similarity).squeeze(-1)
        return self.bceloss(scores, targets)


class ScoreProjectionLoss(nn.Module):
    def __init__(self):
        super(ScoreProjectionLoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, scores_dense, scores_src, proj_pts, invis_idx):
        # scores_src: (Bs, N), scores_dense: (Bd, 1, H, W)
        Bs, Bd, N, _ = proj_pts.shape

        # (Bd, Bs, N)
        scores_dst = F.grid_sample(scores_dense, proj_pts.transpose(0, 1), align_corners=False).squeeze()
        # (Bd, Bs, N)
        scores_src_dup = scores_src[None, :].expand(Bd, Bs, N)

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
