#!/usr/bin/env python3

import torch
import kornia as kn
import torch.nn as nn
import kornia.feature as kf
import torch.nn.functional as F
from utils import Visualization
from utils import PairwiseProjector
import kornia.geometry.conversions as C
from models.featurenet import GridSample


class FeatureNetLoss(nn.Module):
    def __init__(self, beta=[1, 1, 1], K=None, debug=False):
        super().__init__()
        self.beta = beta
        self.sample = GridSample()
        self.distinction = DistinctionLoss()
        self.projector = PairwiseProjector(K)
        self.score_loss = ScoreLoss(debug=debug)
        self.match = DiscriptorMatchLoss(debug=debug)
        self.debug = Visualization('loss') if debug else debug

    def forward(self, descriptors, points, pointness, depths_dense, poses, K, imgs):
        def batch_project(pts):
            return self.projector(pts, depths_dense, poses, K)

        H, W = pointness.size(2), pointness.size(3)
        distinction = self.distinction(descriptors)
        cornerness = self.score_loss(pointness, imgs, batch_project)
        proj_pts, invis_idx = batch_project(points)
        match = self.match(descriptors, points, proj_pts, invis_idx, H, W)

        if self.debug is not False:
            print('Loss: ', distinction, cornerness, match)
            src_idx, dst_idx, pts_idx = invis_idx
            _proj_pts = proj_pts.clone()
            _proj_pts[src_idx, dst_idx, pts_idx, :] = -2
            for dbgpts in _proj_pts:
                self.debug.show(imgs, dbgpts)
            self.debug.showmatch(imgs[0], points[0], imgs[1], proj_pts[0,1])

        return self.beta[0]*distinction + self.beta[1]*cornerness + self.beta[2]*match


class DistinctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.cosine = nn.CosineSimilarity(dim=-2)

    def forward(self, descriptors):
        # batch pairwise cosine
        x = descriptors.permute((1, 2, 0)).unsqueeze(1)
        y = descriptors.permute((1, 2, 0))
        c = self.cosine(x, y)
        pcos = c.permute((2, 0, 1))
        return self.relu(pcos).mean()


class ScoreLoss(nn.Module):
    def __init__(self, radius=8, num_corners=500, debug=False):
        super(ScoreLoss, self).__init__()
        self.radius = radius
        self.bceloss = nn.BCELoss()
        self.corner_det = kf.CornerGFTT()
        self.num_corners = num_corners
        self.debug = Visualization('corners') if debug else debug

    def forward(self, scores_dense, imgs, projector):
        corners = self.get_corners(imgs, projector)

        if self.debug:
            _B = corners.shape[0]
            _coords = corners.squeeze().nonzero(as_tuple=False)
            _pts_list = [_coords[_coords[:, 0] == i][:, [2, 1]] for i in range(_B)]
            _pts = torch.ones(_B, max([p.shape[0] for p in _pts_list]), 2) * -2
            for i, p in enumerate(_pts_list):
                _pts[i, :len(p)] = p
            _pts = C.normalize_pixel_coordinates(_pts, imgs.shape[2], imgs.shape[3])
            self.debug.show(imgs, _pts)

        return self.bceloss(scores_dense, corners)

    def get_corners(self, imgs, projector=None):
        (B, _, H, W), N = imgs.shape, self.num_corners
        corners = kf.nms2d(self.corner_det(kn.rgb_to_grayscale(imgs)), (5, 5))

        # only one in patch
        corners = F.unfold(corners, kernel_size=self.radius, stride=self.radius)
        mask = (corners > 0) & (corners == corners.max(dim=1, keepdim=True).values)
        corners = corners * mask.to(corners)
        corners = F.fold(corners, (H, W), kernel_size=self.radius, stride=self.radius)

        # keep top
        values, idx = corners.view(B, -1).topk(N, dim=1)
        coords = torch.stack([idx % W, idx // W], dim=2)  # (x, y), same below

        if not projector:
            # keep as-is
            b = torch.arange(0, B).repeat_interleave(N).to(idx)
            h, w = idx // W, idx % W
            values = values.flatten()
        else:
            # combine corners from all images
            coords = kn.normalize_pixel_coordinates(coords, H, W)
            coords, invis_idx = projector(coords)
            coords[tuple(invis_idx)] = -2
            coords_combined = coords.transpose(0, 1).reshape(B, B * N, 2)
            coords_combined = kn.denormalize_pixel_coordinates(coords_combined, H, W).round().to(torch.long)
            b = torch.arange(B).repeat_interleave(B * N).to(coords_combined)
            w, h = coords_combined.reshape(-1, 2).T
            mask = w >= 0
            b, h, w, values = b[mask], h[mask], w[mask], values.flatten().repeat(B)[mask]

        target = torch.zeros_like(corners)
        target[b, 0, h, w] = values
        target = kf.nms2d(target, (5, 5))

        return (target > 0).to(target)


class ScoreProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample = GridSample()
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, pointness, scores_src, proj_pts, invis_idx):
        scores_dst = self.sample((pointness, proj_pts))
        scores_src = scores_src.unsqueeze(0).expand_as(scores_dst)
        proj_loss = self.mseloss(scores_dst, scores_src)
        src_idx, dst_idx, pts_idx = invis_idx
        proj_loss[src_idx, dst_idx, pts_idx] = 0
        return proj_loss.mean()


class DiscriptorMatchLoss(nn.Module):
    def __init__(self, radius=1, debug=False):
        super(DiscriptorMatchLoss, self).__init__()
        self.radius, self.debug = radius, debug
        self.cosine = nn.CosineSimilarity()

    def forward(self, descriptors, pts_src, pts_dst, invis_idx, height, width):
        B, N, _ = pts_src.shape

        pts_src = C.denormalize_pixel_coordinates(pts_src.detach(), height, width)
        pts_dst = C.denormalize_pixel_coordinates(pts_dst.detach(), height, width)
        pts_src = pts_src.unsqueeze(0).expand_as(pts_dst).reshape(B**2, N, 2)
        pts_dst = pts_dst.reshape_as(pts_src)

        match = torch.cdist(pts_src, pts_dst)<=self.radius
        invis_bs, invis_bd, invis_n = invis_idx
        match[invis_bs * B + invis_bd, invis_n, :] = 0
        idx = match.triu(diagonal=1).nonzero(as_tuple=True)
        src, dst = [idx[0]%B, idx[1]], [idx[0]//B, idx[2]]
        cosine = self.cosine(descriptors[src], descriptors[dst])

        if self.debug:
            for b, n, b1, n1 in zip(src[0], src[1], dst[0], dst[1]):
                print("%d <-> %d, %d <-> %d (2)" % (b, b1, n, n1))

        return (1 - cosine).mean()
