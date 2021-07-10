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
        depths_rep = src_repeat(depth_map)
        points_rep = src_repeat(points)

        cam_src = self._make_camera(H, W, src_repeat(Ks), src_repeat(poses))
        cam_dst = self._make_camera(H, W, dst_repeat(Ks), dst_repeat(poses))

        proj_p, (pair_idx, point_idx) = self(points_rep, depths_rep, cam_src, cam_dst, depths_rep)
        return proj_p.reshape(B, B, N, 2), torch.stack([pair_idx // B, pair_idx % B, point_idx])

    def pix2world(self, points, depth_map, poses, Ks):
        """Unprojects pixels to 3D coordinates."""
        H, W = depth_map.shape[2:]
        cam = self._make_camera(H, W, Ks, poses)
        depths = self._sample_depths(depth_map, points)
        return self._pix2world(points, depths, cam)

    def world2pix(self, points, res, poses, Ks, depth_map=None, eps=1e-2):
        """Projects 3D coordinates to screen."""
        cam = self._make_camera(res[0], res[1], Ks, poses)
        xy, depth = self._world2pix(points, cam)

        if depth_map is not None:
            depth_dst = self._sample_depths(depth_map, xy)
            xy[(depth < 0) | (depth > depth_dst + eps) | ((xy.abs() > 1).any(dim=-1))] = np.nan

        return xy, depth

    def bounding_frustum(self, points, depth_map, poses, Ks):
        """Finds the viewing frustum (corners and face equations) bounding the points."""
        B, _, H, W = depth_map.shape
        depths = self._sample_depths(depth_map, points)
        assert depths.isfinite().all()

        min_depth, max_depth = depths.min(dim=1).values, depths.max(dim=1).values
        vert_locs = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]).repeat(B, 2, 1).to(depths)
        vert_depths = torch.cat([min_depth.repeat(4, 1), max_depth.repeat(4, 1)], dim=0).T

        cam = self._make_camera(H, W, Ks, poses)
        verts = self._pix2world(vert_locs, vert_depths, cam)
        faces = torch.tensor([[0, 2, 3], [2, 7, 3], [1, 3, 5],
                              [0, 1, 4], [0, 6, 2], [4, 5, 7]]).to(verts.device)

        # (3 [triangle verts], 6 [faces], B, 3 [xyz])
        p, q, r = verts.transpose(0, 1)[faces.T]
        # (B, 6 [faces], 3 [xyz])
        abc = torch.cross(r - p, q - p, dim=-1).transpose(0, 1)
        d = -torch.einsum('bfp,bfp->bf', abc, p.transpose(0, 1)).unsqueeze(-1)
        eqn = torch.cat([abc, d], -1)
        # print(eqn)
        return verts, eqn

    def frustum_intersects(self, verts0, eqn1):
        B0, N, _ = verts0.shape
        verts0_h = torch.cat([verts0, torch.ones(B0, N, 1).to(verts0)], dim=-1)
        # print(torch.einsum('anp,bfp->abnf', verts0_h, eqn1))
        outside = torch.einsum('anp,bfp->abnf', verts0_h, eqn1) > 0
        # print(outside)
        # diff_side = side.sum(2).abs() < N # ax + by + cz + d has different sign for any point?
        # print(diff_side)
        return ~outside.all(-2).any(-1)
        

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

def feature_pt_ncovis(pos0, pts1, depth1, pose1, K1, projector, grid_sample, eps=1e-2, ret_proj=False, grid_size=(12, 16)):
    B0, B1 = len(pos0), len(pts1)
    _, _, H, W = depth1.shape

    # find where points from other frames land
    pts0_scr1, pts0_depth1 = projector.world2pix(src_repeat(pos0, B1), (H, W),
        dst_repeat(pose1, B0), dst_repeat(K1, B0))
    _, N, _ = pts0_scr1.shape
    pts0_scr1_depth1 = grid_sample((depth1, pts0_scr1.reshape(B0, B1, N, 2).transpose(0, 1).reshape(B1, B0 * N, 2))).reshape(B1, B0, N).transpose(0, 1).reshape(B0 * B1, N)
    pts0_scr1[(pts0_depth1 < 0) | (pts0_depth1 > pts0_scr1_depth1 + eps) | ((pts0_scr1.abs() > 1).any(dim=-1))] = np.nan

    Ax = pts0_scr1.isfinite().all(dim=-1).to(torch.float).mean(dim=-1)
    ########
    # plt.figure()
    # plt.scatter(*pts0_scr1.cpu().numpy().T)
    # plt.gca().invert_yaxis()
    ########
    binned_pts0_scr1 = kn.denormalize_pixel_coordinates(pts0_scr1, *grid_size).round()
    # binning
    B, N, _ = binned_pts0_scr1.shape
    bs = torch.arange(B).repeat_interleave(N).unsqueeze(1)
    bp0s1_b = torch.cat([bs.to(binned_pts0_scr1), binned_pts0_scr1.reshape(-1, 2)], axis=-1)
    bp0s1_b = bp0s1_b[bp0s1_b.isfinite().all(dim=-1)]

    A1 = torch.zeros(B).to(Ax)
    if bp0s1_b.numel() != 0:
        # count unique (b, x, y) for each b
        bs, count = bp0s1_b.unique(dim=0)[:, 0].unique_consecutive(return_counts=True)
        
        A1[bs.to(torch.long)] = count / np.product(grid_size)
        ########
        # plt.figure()
        # plt.scatter(*binned_pts0_scr1.cpu().numpy().T)
        # plt.gca().invert_yaxis()
        ########
        # print([binned[binned.isfinite().all(dim=-1)].unique(dim=0) / np.product(grid_size) for binned in binned_pts0_scr1])
        # A1 = torch.tensor([len(binned[binned.isfinite().all(dim=-1)].unique(dim=0)) / np.product(grid_size) for binned in binned_pts0_scr1]).to(Ax)

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
