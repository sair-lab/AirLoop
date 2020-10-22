#!/usr/bin/env python3

import time
import torch


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.duration = time.time()-self.start_time
        self.start()
        return self.duration


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pix2world(p, depth, T_p, K_inv):
    """Transforms from pixel coordinates to world frame.

    Args:
      p:     Pixel coordinates (N, 2).
      depth: Depth of each point (N).
      T_p:   Camera poses in which p is observed (N, 3, 4).
      K_inv: Inverse of camera intrinsics (N, 3, 3).

    Returns:
      Coordinates in world frame (N, 3).
    """
    N = len(p)

    p_h = torch.cat([p, torch.ones(N, 1, device=p.get_device())], 1).unsqueeze(2)
    p_cam = torch.bmm(K_inv, p_h) * depth.reshape(N, 1, 1)
    # T_p^-1 * p_cam
    R, t = T_p[:, :, :3], T_p[:, :, 3].unsqueeze(2)
    p_world_h = torch.bmm(R.transpose(1, 2), p_cam - t).squeeze(2)
    return p_world_h


def world2pix(p, T_p, K):
    """Transforms world frame to pixel coordinates.

    Args:
      p:   World coordinates (N, 3).
      T_p: Camera poses in which p is observed (N, 3, 4).
      K:   Camera intrinsics (N, 3, 3).

    Returns:
      Pixel corrdinates (N, 2).
    """
    N = len(p)

    p_h = torch.cat([p, torch.ones(N, 1, device=p.get_device())], 1).unsqueeze(2)
    p_cam_h = torch.bmm(T_p, p_h)
    pix_h = torch.bmm(K, p_cam_h).squeeze(2)
    return pix_h[:, :2] / pix_h[:, 2].unsqueeze(1)


def project_points(p, depth, T_p, T_q, K, K_inv=None):
    """Projects p visible in pose T_p to pose T_q.

    Args:
      p:     List of points (N, 2).
      depth: Depth of each point(N).
      T_p:   List of camera poses in which p is observed (N, 3, 4).
      T_q:   List of camera poses to project into (N, 3, 4).
      K:     Camera intrinsics (3, 3).
      K_inv: Optional; precomputed inverse of K (3, 3).

    Returns:
      Coordinates of p in pose T_q (N, 2).
    """
    world_coord = pix2world(p, depth, T_p, K_inv if K_inv else torch.inverse(K))
    return world2pix(world_coord, T_q, K)
