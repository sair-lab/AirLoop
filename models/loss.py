#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import project_points


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
