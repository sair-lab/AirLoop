#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import project_points

# TEMP
import cv2
from utils import Visualization
vis = Visualization('src->tgt')
# TEMP


class FeatureNetLoss(nn.Module):
    def __init__(self, lamb_dist=1):
        super().__init__()
        self.distinctloss = DistinctivenessLoss()
        self.lamb_dist = lamb_dist

    def forward(self, features, points, scores_dense, depths_dense, poses, K, K_inv, imgs):
        scores = [scores_dense[b, 0, p[:, 0], p[:, 1]]
                  for b, p in enumerate(points)]
        depths = [depths_dense[b, 0, p[:, 0], p[:, 1]]
                  for b, p in enumerate(points)]

        total_loss = self.lamb_dist * self.distinctloss(features, scores)

        return total_loss


class DistinctivenessLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(DistinctivenessLoss, self).__init__()
        self.bceloss = nn.BCELoss()
        self.eps = eps

    def forward(self, features, scores):
        score_dist_loss = 0
        for feature, score in zip(features, scores):
            feature = F.normalize(feature, eps=self.eps)
            sum_feature = feature.sum(dim=0)
            # ave_cos_sim_i = (sum_{j != i} f_i' * f_j) / (n - 1) = ((f_i' * sum_j f_j) - 1) / (n - 1)
            ave_cossim = (feature @ sum_feature.T - 1) / (len(feature) - 1)
            score_dist_loss += self.bceloss(score,
                                            (1 - F.relu(ave_cossim)).detach())

        return score_dist_loss


# Deprecate
def training_criterion(features, points, scores_dense, depths_dense, poses, K, K_inv, imgs):
    img_boudnary = torch.tensor(
        [[0, 0], torch.tensor(scores_dense.shape[2:4]) - 1]).to(scores_dense)

    score_proj_loss = 0
    # project points in every frame to every other frame
    for src, (src_pts, src_d, src_T, src_scores_all) in enumerate(zip(points, depths, poses, scores)):
        # TEMP
        dbgpts = [src_pts]
        images = [imgs[src]]
        # TEMP
        for tgt in range(len(points)):
            if src == tgt:
                continue

            tgt_scores_dense = scores_dense[tgt]
            tgt_T = poses[tgt]
            tgt_pts, covis_idx = project_points(
                src_pts, src_d, src_T, tgt_T, K, K_inv, trim_boundary=img_boudnary)

            # TEMP
            dbgpts.append(tgt_pts)
            print(tgt)
            images.append(imgs[tgt])
            # TEMP

            tgt_pts = (2 * tgt_pts / img_boudnary[1] - 1).view(1, 1, -1, 2)
            tgt_scores = F.grid_sample(
                tgt_scores_dense.unsqueeze(0), tgt_pts, align_corners=True).squeeze()

            score_proj_loss += F.mse_loss(src_scores_all[covis_idx], tgt_scores)

        # TEMP
        vis.show(torch.stack(images), dbgpts)
        while cv2.waitKey() != ord('n'):
            pass
        # TEMP

    return score_dist_loss, score_proj_loss


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


def descriptor_loss(D_p, D_q, is_inlier, red='none'):
    """Computes loss on discriptors from matcher.

    Args:
      D_p, D_q: The descriptors  (N, D).
      is_inlier: Is this pair a inliner (N)?
      red:       Optional. Reduce to one number?

    Returns:
      cos_sim if is_inlier else max(cos_sim, 0).
    """
    return F.cosine_embedding_loss(D_p, D_q, is_inlier, reduction=red)
