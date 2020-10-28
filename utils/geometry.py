#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation as R

from kornia.geometry import PinholeCamera, DepthWarper, denormalize_pixel_coordinates


class PairwiseProjector(nn.Module):
    def __init__(self, width, height, K=None):
        super(PairwiseProjector, self).__init__()

        self.width, self.height = width, height
        self.K = K

    def forward(self, points, depths_dense, poses, K=None, occ_idx=True):
        if K is not None:
            self.K = K

        B, N = points.shape[:2]

        # TODO don't self-project

        # [T0 T0 ... T1 T1 ...] vs [T0 T1 ... T0 T1 ...]
        poses_src = poses.repeat_interleave(B, dim=0)
        poses_dst = poses.repeat(B, 1, 1)

        # (B, N, 2)
        points_px = denormalize_pixel_coordinates(
            points, self.height, self.width)
        # (B, N)
        depths = F.grid_sample(
            depths_dense, points[:, None], align_corners=False).squeeze()

        # (B**2, N, 2)
        points_px_dup = self._batch_repeat(points_px)
        # (B**2, N)
        depths_dup = self._batch_repeat(depths)

        cam_src = make_camera(self.width, self.height, self.K, poses_src, B**2)
        cam_dst = make_camera(self.width, self.height, self.K, poses_dst, B**2)

        # (B**2, N, 2)
        proj_p = project_points(points_px_dup, depths_dup,
                                cam_src, cam_dst)

        if occ_idx:
            in_boundary = torch.all((proj_p >= -1) & (proj_p <= 1), dim=-1)

            depths_dense_dst = self._batch_repeat(depths_dense)
            depths_dst = F.grid_sample(depths_dense_dst, proj_p[:, None],
                                       align_corners=False).squeeze()
            # TODO compare with projected depth

            visible_idx = torch.nonzero(in_boundary, as_tuple=True)

            return proj_p.reshape(B, B, N, 2), visible_idx
        else:
            return proj_p.reshape(B, B, N, 2)

    @staticmethod
    def _batch_repeat(x):
        """Duplicate along batch for batch_size times"""
        B, shape = x.shape[0], x.shape[1:]
        return x[:, None].expand(B, B, *shape).reshape(B**2, *shape)


def pose2mat(pose):
    """Converts pose vectors to matrices.

    Args:
      pose: [tx, ty, tz, qx, qy, qz, qw] (N, 7).

    Returns:
      [R t] (N, 3, 4).
    """
    t = pose[:, 0:3, None]
    rot = R.from_quat(pose[:, 3:7]).as_matrix().astype(np.float32).transpose(0, 2, 1)
    t = -rot @ t
    return torch.cat([torch.from_numpy(rot), torch.from_numpy(t)], dim=2)


def make_camera(width, height, K, pose, batch_size=1):
    """Creates a PinholeCamera with specified info"""
    intrinsics = torch.eye(4, 4).to(K).repeat(batch_size, 1, 1)
    intrinsics[:, 0:3, 0:3] = K

    extrinsics = torch.eye(4, 4).to(K).repeat(batch_size, 1, 1)
    extrinsics[:, 0:3, 0:4] = pose

    height, width = torch.tensor([height]).to(K), torch.tensor([width]).to(K)

    return PinholeCamera(intrinsics, extrinsics, height, width)


def project_points(p, depth_src, cam_src, cam_dst):
    """Projects p visible in pose T_p to pose T_q.

    Args:
      p:                List of points in pixels (B, N, 2).
      depth:            Depth of each point(B, N).
      cam_src, cam_dst: Source and destination cameras with batch size B

    Returns:
      Normalized coordinates of p in pose cam_dst (N, 2).
    """
    b, n, _ = p.shape
    assert b == cam_src.batch_size == cam_dst.batch_size

    warper = DepthWarper(cam_dst,
                         int(cam_dst.height.item()),
                         int(cam_dst.width.item())).to(p)
    # (B, 1, N, 3)
    warper.grid = torch.cat([p, torch.ones(b, n, 1).to(p)], dim=2)[:, None]
    warper.compute_projection_matrix(cam_src)
    projected = warper.warp_grid(depth_src.view(b, 1, 1, n)).squeeze()
    return projected
