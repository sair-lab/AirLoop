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


def training_criterion(features, points, scores_dense, depths_dense, poses, K, K_inv, imgs):
    # TODO distinciveness loss

    scores = [scores_dense[src, 0, p[:, 0], p[:, 1]]
              for src, p in enumerate(points)]
    depths = [depths_dense[src, 0, p[:, 0], p[:, 1]]
              for src, p in enumerate(points)]

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


def score_loss(score, is_inlier, red='none'):
    """Computes loss on confidence scores from matcher.

    Args:
      score:     The score between 0 and 1 of matching point pairs from matcher (N).
      is_inlier: Is this pair a inliner (N)?
      red:       Optional. Reduce to one number?

    Returns:
      CrossEntropy(score, e_pq <= in_thresh) (N).
    """
    return F.binary_cross_entropy(score, is_inlier, reduction=red)


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
