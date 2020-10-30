#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Visualization
from utils import PairwiseProjector
from models.featurenet import GridSample
from kornia import PinholeCamera, DepthWarper, denormalize_pixel_coordinates


class GridMatch(nn.Module):
    def __init__(self):
        super(GridMatch, self).__init__()

    def forward(self, p_src, p_dst, height, width, grid_scale):
        grid_h, grid_w = int(height * grid_scale), int(width * grid_scale)
        # index along Bd, Bs, N
        visible_src_idx = self._get_visible_idx(p_src.detach())
        visible_dst_idx = self._get_visible_idx(p_dst.detach())

        src_coord = denormalize_pixel_coordinates(p_src[tuple(visible_src_idx)], grid_h, grid_w)
        dst_coord = denormalize_pixel_coordinates(p_dst[tuple(visible_dst_idx)], grid_h, grid_w)
        xs, ys = src_coord.to(torch.long).T
        xd, yd = dst_coord.to(torch.long).T

        # register points in grid of indices into valid_idx*, (Bd, Bs, x, y, 0/1)
        grid = -torch.ones(*p_src.shape[:2], grid_w, grid_h, 2).to(p_src.device, torch.long)
        grid[visible_src_idx[0], visible_src_idx[1], xs, ys, 0] = torch.arange(len(xs)).to(xs)
        grid[visible_dst_idx[0], visible_dst_idx[1], xd, yd, 1] = torch.arange(len(xd)).to(xd)

        match_src_idx, match_dst_idx = grid[(grid[..., 0] >= 0) & (grid[..., 1] >= 0)].T

        # restore to image pixel size
        factor = torch.tensor([(height - 1) / (grid_h - 1), (width - 1) / (grid_w - 1)]).to(src_coord)
        src_coord, dst_coord = src_coord * factor, dst_coord * factor

        # pixel coordinates and index along Bs, N and Bd, N
        return src_coord[match_src_idx], dst_coord[match_dst_idx], \
            visible_src_idx[:, match_src_idx][[1, 2]].T, visible_dst_idx[:, match_dst_idx][[0, 2]].T

    def _get_visible_idx(self, pts):
        return torch.nonzero(torch.all((pts >= -1) & (pts <= 1), dim=-1), as_tuple=False).T


class FeatureNetLoss(nn.Module):
    def __init__(self, alpha=[1, 1, 1], K=None, debug=False):
        super().__init__()
        self.alpha = alpha
        self.sample = GridSample()
        self.distinction = DistinctionLoss()
        self.projection = ScoreProjectionLoss()
        self.projector = PairwiseProjector(K)
        self.match = DiscriptorMatchLoss(debug=debug)
        self.debug = Visualization('loss') if debug else debug

    def forward(self, features, points, scores_dense, depths_dense, poses, K, imgs):
        scores = self.sample((scores_dense, points))
        distinction = self.distinction(features, scores)
        proj_pts, invis_idx = self.projector(points, depths_dense, poses, K)
        projection = self.projection(scores_dense, scores, proj_pts, invis_idx)
        match = self.match(features, points, proj_pts, invis_idx, *scores_dense.shape[2:4])

        if self.debug is not False:
            src_idx, dst_idx, pts_idx = invis_idx
            _proj_pts = proj_pts.clone()
            _proj_pts[src_idx, dst_idx, pts_idx, :] = -2
            for dbgpts in _proj_pts:
                self.debug.show(imgs, dbgpts)

        return self.alpha[0] * distinction + self.alpha[1] * projection + self.alpha[2] * match


class DistinctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.bceloss = nn.BCELoss()

    def forward(self, features, scores):
        features = F.normalize(features, dim=1).detach()
        summation = features.sum(dim=1, keepdim=True).transpose(1, 2)
        similarity = (features@summation - 1)/(features.size(1) - 1)
        targets = 1 - self.relu(similarity)
        return self.bceloss(scores, targets)


class ScoreProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample = GridSample()
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, scores_dense, scores_src, proj_pts, invis_idx):
        scores_dst = self.sample((scores_dense, proj_pts))
        scores_src = scores_src.unsqueeze(1).expand_as(scores_dst)
        proj_loss = self.mseloss(scores_dst, scores_src)
        src_idx, dst_idx, pts_idx = invis_idx
        proj_loss[src_idx, dst_idx, pts_idx] = 0
        return proj_loss.mean()


class DiscriptorMatchLoss(nn.Module):
    def __init__(self, radius_thresh=1, debug=False):
        super(DiscriptorMatchLoss, self).__init__()
        self.match = GridMatch()
        self.similarity = nn.CosineSimilarity()
        self.thresh = radius_thresh
        self.debug = debug

    def forward(self, features, points, proj_pts, invis_idx, height, width):
        # TEMP
        import time
        start = time.time()
        for _ in range(1000):
            # match_loss = self.forward1(features, points, proj_pts, invis_idx, height, width)
            match_loss = self.forward2(features, points, proj_pts, invis_idx, height, width)
        print((time.time() - start) / 1000)
        # TEMP

        return match_loss

    def forward1(self, features, points, proj_pts, invis_idx, height, width, scale=0.5):
        points_coord, proj_pts_coord, points_b_n, proj_pts_b_n = \
            self.match(points.unsqueeze(1).expand_as(proj_pts), proj_pts, height, width, scale)

        dist = F.pairwise_distance(points_coord, proj_pts_coord)
        match_idx = torch.nonzero(dist <= self.thresh / 2, as_tuple=True)

        points_b_n, proj_pts_b_n = tuple(points_b_n[match_idx].T), tuple(proj_pts_b_n[match_idx].T)
        if self.debug:
            for b, n, b1, n1 in zip(points_b_n[0], points_b_n[1], proj_pts_b_n[0], proj_pts_b_n[1]):
                print("%d <-> %d, %d <-> %d (1)" % (b, b1, n, n1))
        match_loss = (1 - self.similarity(features[points_b_n], features[proj_pts_b_n])).sum()

        return match_loss

    def forward2(self, features, points, proj_pts, invis_idx, height, width):
        B, N, _ = points.shape

        points = denormalize_pixel_coordinates(points.detach(), height, width)
        proj_pts = denormalize_pixel_coordinates(proj_pts.detach(), height, width)

        points = points.unsqueeze(1).expand_as(proj_pts).reshape(B**2, N, 2)
        proj_pts = proj_pts.reshape_as(points)

        dist = torch.cdist(points, proj_pts)
        match_idx = torch.nonzero((dist <= self.thresh).triu(), as_tuple=True)

        points_b_n, proj_pts_b_n = [match_idx[0] % B, match_idx[1]], [match_idx[0] // B, match_idx[2]]
        if self.debug:
            for b, n, b1, n1 in zip(points_b_n[0], points_b_n[1], proj_pts_b_n[0], proj_pts_b_n[1]):
                print("%d <-> %d, %d <-> %d (2)" % (b, b1, n, n1))
        match_loss = (1 - self.similarity(features[points_b_n], features[proj_pts_b_n])).sum()

        return match_loss
