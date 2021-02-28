#!/usr/bin/env python3

import torch
import numpy as np
import kornia as kn
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, eps=1e-2, max_depth=1000):
        super().__init__()
        self.eps, self.max_depth = eps, max_depth

    def forward(self, points, depth_map, cam_src, cam_dst, depth_map_dst=None):
        B, N, _ = points.shape
        assert B == len(depth_map) == cam_src.batch_size == cam_dst.batch_size

        depths = self._sample_depths(depth_map, points)
        proj_p, proj_depth = self._project_points(points, depths, cam_src, cam_dst)

        # check for occlusion
        is_out_of_bound = torch.any((proj_p < -1) | (proj_p > 1), dim=-1)

        depths_dst = self._sample_depths(depth_map_dst, proj_p) if depth_map_dst is not None else self.max_depth
        is_occluded = proj_depth.isnan() | (proj_depth > depths_dst + self.eps) | (depths_dst > self.max_depth)

        return proj_p, torch.nonzero(is_out_of_bound | is_occluded, as_tuple=True)

    def cartesian(self, points, depth_map, poses, Ks):
        """Projects points from every view to every other view."""
        B, N = points.shape[:2]
        H, W = depth_map.shape[2:]
        # TODO don't self-project
        # [0 0 0 ... 1 1 1 ... 2 2 2 ...] vs [0 1 2 ... 0 1 2 ... 0 1 2 ...]
        depths_rep = self._src_repeat(depth_map)
        points_rep = self._src_repeat(points)

        cam_src = self._make_camera(H, W, self._src_repeat(Ks), self._src_repeat(poses))
        cam_dst = self._make_camera(H, W, self._dst_repeat(Ks), self._dst_repeat(poses))

        proj_p, (pair_idx, point_idx) = self(points_rep, depths_rep, cam_src, cam_dst, depths_rep)
        return proj_p.reshape(B, B, N, 2), torch.stack([pair_idx // B, pair_idx % B, point_idx])

    def pix2world(self, points, depth_map, poses, Ks):
        """Unprojects pixels to 3D coordinates."""
        H, W = depth_map.shape[2:]
        cam = self._make_camera(H, W, Ks, poses)
        depths = self._sample_depths(depth_map, points)
        return self._pix2world(points, depths, cam)

    def world2pix(self, points, res, poses, Ks):
        """Projects 3D coordinates to screen."""
        cam = self._make_camera(res[0], res[1], Ks, poses)
        return self._world2pix(points, cam)

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

    @staticmethod
    def _src_repeat(x):
        """[b0 b1 b2 ...] -> [b0 b0 ... b1 b1 ...]"""
        B, shape = x.shape[0], x.shape[1:]
        return x.unsqueeze(1).expand(B, B, *shape).reshape(B**2, *shape)

    @staticmethod
    def _dst_repeat(x):
        """[b0 b1 b2 ...] -> [b0 b1 ... b0 b1 ...]"""
        B, shape = x.shape[0], x.shape[1:]
        return x.unsqueeze(0).expand(B, B, *shape).reshape(B**2, *shape)
