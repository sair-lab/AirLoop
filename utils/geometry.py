#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn
import kornia.geometry as G
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


class PairwiseProjector(nn.Module):
    def __init__(self, K=None, eps=1e-2):
        super().__init__()
        self.K, self.eps = K, eps

    def forward(self, points, depths_dense, poses, K=None, ret_invis_idx=True):
        if K is not None:
            self.K = K

        B, N = points.shape[:2]
        H, W = depths_dense.shape[2:]

        # TODO don't self-project

        # [T0 T0 ... T1 T1 ...] vs [T0 T1 ... T0 T1 ...]
        poses_src = poses.repeat_interleave(B, dim=0)
        poses_dst = poses.repeat(B, 1, 1)

        # (B, N)
        depths = F.grid_sample(depths_dense, points[:, None], align_corners=False).squeeze(1).squeeze(1)

        # (B, N, 2)
        points_px = G.denormalize_pixel_coordinates(points, H, W)


        # (B**2, N, 2)
        points_px_dup = self._batch_repeat(points_px)
        # (B**2, N)
        depths_dup = self._batch_repeat(depths)

        cam_src = make_camera(H, W, self.K, poses_src, B**2)
        cam_dst = make_camera(H, W, self.K, poses_dst, B**2)

        # (B**2, N, 2)
        proj_p, proj_depth = project_points(points_px_dup, depths_dup, cam_src, cam_dst)

        if ret_invis_idx:
            is_out_of_bound = torch.all((proj_p < -1) & (proj_p > 1), dim=-1)

            H, W = depths_dense.shape[2:4]
            depths_dense_dup = depths_dense.expand(B, B, H, W).transpose(0, 1).reshape(B**2, 1, H, W)
            depths_dst = F.grid_sample(depths_dense_dup, proj_p[:, None], align_corners=False).squeeze(1).squeeze(1)
            is_occluded = proj_depth > depths_dst + self.eps

            pair_idx, point_idx = torch.nonzero(is_out_of_bound | is_occluded, as_tuple=True)

            invis_idx = torch.stack([pair_idx // B, pair_idx % B, point_idx])

            return proj_p.reshape(B, B, N, 2), invis_idx
        else:
            return proj_p.reshape(B, B, N, 2)

    @staticmethod
    def _batch_repeat(x):
        """[b0 b1 b2 ...] -> [b0 b0 ... b1 b1 ...]"""
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


def make_camera(height, width, K, pose, batch_size=1):
    """Creates a PinholeCamera with specified info"""
    intrinsics = torch.eye(4, 4).to(K).repeat(batch_size, 1, 1)
    intrinsics[:, 0:3, 0:3] = K.repeat(K.size(0),1,1)

    extrinsics = torch.eye(4, 4).to(K).repeat(batch_size, 1, 1)
    extrinsics[:, 0:3, 0:4] = pose

    height, width = torch.tensor([height]).to(K), torch.tensor([width]).to(K)

    return G.PinholeCamera(intrinsics, extrinsics, height, width)


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

    warper = G.DepthWarper(cam_dst, int(cam_dst.height.item()), int(cam_dst.width.item())).to(p)
    warper.compute_projection_matrix(cam_src)
    # (B, 1, N, 3)
    pix_coords = torch.cat([p, torch.ones(b, n, 1).to(p)], dim=2)[:, None]
    projected, proj_depth = warp_grid(warper, pix_coords, depth_src.view(b, 1, 1, n))
    return projected.squeeze(1), proj_depth.squeeze(1)


def warp_grid(warper, pixel_coords, depth_src):
    """Adapted from kornia.geometry.warp.depth_warper."""
    # unpack depth attributes
    dtype = depth_src.dtype

    # reproject the pixel coordinates to the camera frame
    cam_coords_src = G.pixel2cam(
        depth_src,
        warper._pinhole_src.intrinsics_inverse().to(dtype),
        pixel_coords)  # BxHxWx3

    # reproject the camera coordinates to the pixel
    pixel_coords_src, z_coord = _cam2pixel(
        cam_coords_src, warper._dst_proj_src.to(dtype))  # (B*N)xHxWx2

    # normalize between -1 and 1 the coordinates
    pixel_coords_src_norm = G.normalize_pixel_coordinates(
        pixel_coords_src, warper.height, warper.width)

    return pixel_coords_src_norm, z_coord


def _cam2pixel(cam_coords_src, dst_proj_src, eps=1e-6):
    """Adapted from kornia.geometry.camera.pinhole."""
    # apply projection matrix to points
    point_coords = G.transform_points(dst_proj_src[:, None], cam_coords_src)
    x_coord = point_coords[..., 0]
    y_coord = point_coords[..., 1]
    z_coord = point_coords[..., 2]

    # compute pixel coordinates
    u_coord = x_coord / (z_coord + eps)
    v_coord = y_coord / (z_coord + eps)

    # stack and return the coordinates, that's the actual flow
    pixel_coords_dst = torch.stack([u_coord, v_coord], dim=-1)
    return pixel_coords_dst, z_coord  # (B*N)xHxWx2
