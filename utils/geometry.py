#!/usr/bin/env python3

import kornia as kn
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from .utils import coord_list_grid_sample


class Projector():

    @staticmethod
    def pix2world(points, depth_map, poses, Ks):
        """Unprojects pixels to 3D coordinates."""
        H, W = depth_map.shape[2:]
        cam = Projector._make_camera(H, W, Ks, poses)
        depths = Projector._sample_depths(depth_map, points)
        return Projector._pix2world(points, depths, cam)

    @staticmethod
    def world2pix(points, res, poses, Ks, depth_map=None, eps=1e-2):
        """Projects 3D coordinates to screen."""
        cam = Projector._make_camera(res[0], res[1], Ks, poses)
        xy, depth = Projector._world2pix(points, cam)

        if depth_map is not None:
            depth_dst = Projector._sample_depths(depth_map, xy)
            xy[(depth < 0) | (depth > depth_dst + eps) | ((xy.abs() > 1).any(dim=-1))] = np.nan

        return xy, depth

    @staticmethod
    def _make_camera(height, width, K, pose):
        """Creates a PinholeCamera with specified intrinsics and extrinsics."""
        intrinsics = torch.eye(4, 4).to(K).repeat(len(K), 1, 1)
        intrinsics[:, 0:3, 0:3] = K

        extrinsics = torch.eye(4, 4).to(pose).repeat(len(pose), 1, 1)
        extrinsics[:, 0:3, 0:4] = pose

        height, width = torch.tensor([height]).to(K), torch.tensor([width]).to(K)

        return kn.PinholeCamera(intrinsics, extrinsics, height, width)

    @staticmethod
    def _pix2world(p, depth, cam):
        """Projects p to world coordinate.

        Args:
        p:     List of points in pixels (B, N, 2).
        depth: Depth of each point(B, N).
        cam:   Camera with batch size B

        Returns:
        World coordinate of p (B, N, 3).
        """
        p = kn.denormalize_pixel_coordinates(p, int(cam.height), int(cam.width))
        p_h = kn.convert_points_to_homogeneous(p)
        p_cam = kn.transform_points(cam.intrinsics_inverse(), p_h) * depth.unsqueeze(-1)
        return kn.transform_points(kn.inverse_transformation(cam.extrinsics), p_cam)

    @staticmethod
    def _world2pix(p_w, cam):
        """Projects p to normalized camera coordinate.

        Args:
        p_w: List of points in world coordinate (B, N, 3).
        cam: Camera with batch size B

        Returns:
        Normalized coordinates of p in pose cam_dst (B, N, 2) and screen depth (B, N).
        """
        proj = kn.compose_transformations(cam.intrinsics, cam.extrinsics)
        p_h = kn.transform_points(proj, p_w)
        p, d = kn.convert_points_from_homogeneous(p_h), p_h[..., 2]
        return kn.normalize_pixel_coordinates(p, int(cam.height), int(cam.width)), d

    @staticmethod
    def _project_points(p, depth_src, cam_src, cam_dst):
        """Projects p visible in pose T_p to pose T_q.

        Args:
        p:                List of points in pixels (B, N, 2).
        depth:            Depth of each point(B, N).
        cam_src, cam_dst: Source and destination cameras with batch size B

        Returns:
        Normalized coordinates of p in pose cam_dst (B, N, 2).
        """
        return Projector._world2pix(Projector._pix2world(p, depth_src, cam_src), cam_dst)

    @staticmethod
    def _sample_depths(depths_map, points):
        """Samples the depth of each point in points"""
        assert depths_map.shape[:2] == (len(points), 1)
        return F.grid_sample(depths_map, points[:, None], align_corners=False)[:, 0, 0, ...]

def src_repeat(x, n_dst=None):
    """[b0 b1 b2 ...] -> [b0 b0 ... b1 b1 ...]"""
    B, shape = x.shape[0], x.shape[1:]
    n_dst = n_dst if n_dst is not None else B
    return x.unsqueeze(1).expand(B, n_dst, *shape).reshape(B * n_dst, *shape)

def dst_repeat(x, n_src=None):
    """[b0 b1 b2 ...] -> [b0 b1 ... b0 b1 ...]"""
    B, shape = x.shape[0], x.shape[1:]
    n_src = n_src if n_src is not None else B
    return x.unsqueeze(0).expand(n_src, B, *shape).reshape(n_src * B, *shape)

# from matplotlib import pyplot as plt

def feature_pt_ncovis(pos0, pts1, depth1, pose1, K1, eps=1e-2, ret_proj=False, grid_size=(12, 16)):
    B0, B1 = len(pos0), len(pts1)
    _, _, H, W = depth1.shape

    # find where points from other frames land
    pts0_scr1, pts0_depth1 = Projector.world2pix(src_repeat(pos0, B1), (H, W),
        dst_repeat(pose1, B0), dst_repeat(K1, B0))
    _, N, _ = pts0_scr1.shape
    pts0_scr1_depth1 = coord_list_grid_sample(depth1, pts0_scr1.reshape(B0, B1, N, 2).transpose(0, 1).reshape(B1, B0 * N, 2)).reshape(B1, B0, N).transpose(0, 1).reshape(B0 * B1, N)
    pts0_scr1[(pts0_depth1 < 0) | (pts0_depth1 > pts0_scr1_depth1 + eps) | ((pts0_scr1.abs() > 1).any(dim=-1))] = np.nan

    Ax = pts0_scr1.isfinite().all(dim=-1).to(torch.float).mean(dim=-1)
    binned_pts0_scr1 = kn.denormalize_pixel_coordinates(pts0_scr1, *grid_size).round()
    # binning
    B, N, _ = binned_pts0_scr1.shape
    valid_b, valid_n = binned_pts0_scr1.isfinite().all(dim=-1).nonzero(as_tuple=True)
    bp0s1_b = torch.cat([valid_b[:, None], binned_pts0_scr1[valid_b, valid_n]], axis=-1)

    A1 = torch.zeros(B).to(Ax)
    if bp0s1_b.numel() != 0:
        # count unique (b, x, y) for each b
        bs, count = bp0s1_b.unique(dim=0)[:, 0].unique_consecutive(return_counts=True)
        
        A1[bs.to(torch.long)] = count / np.product(grid_size)

    covis = Ax / (1 + Ax / A1.clamp(min=1e-6) - Ax)

    if ret_proj:
        return covis.reshape(B0, B1), pts0_scr1.reshape(B0, B1, N, 2)
    return covis.reshape(B0, B1)


def gen_probe(depth_map, scale=8):
    B, _, H, W = depth_map.shape
    h, w = H // scale, W // scale
    points = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=2) + 0.5
    points = kn.normalize_pixel_coordinates(points, w + 1, h + 1).unsqueeze(0).expand(B, -1, -1, -1)
    return points.reshape(B, -1, 2).to(depth_map)


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
