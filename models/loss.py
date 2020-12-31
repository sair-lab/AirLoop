#!/usr/bin/env python3

import torch
import kornia as kn
import torch.nn as nn
import kornia.feature as kf
import torch.nn.functional as F
import kornia.geometry.conversions as C

from utils import Projector
from utils import Visualizer
from models.memory import Memory
from models.featurenet import GridSample
from models.BAnet import ConsecutiveMatch

class FeatureNetLoss(nn.Module):
    def __init__(self, beta=[1, 1, 1, 10], K=None, debug=False, writer=None, viz_start=float('inf'), viz_freq=200):
        super().__init__()
        self.writer, self.beta, self.n_iter = writer, beta, 0
        self.score_corner = ScoreLoss(debug=debug)
        self.desc_dist = DiscriptorDistinctionLoss()
        self.desc_match = DiscriptorMatchLoss(writer=writer)
        self.desc_ret = DiscriptorRententionLoss(writer=writer)
        self.projector = Projector()
        self.viz = Visualizer() if self.writer is None else Visualizer('tensorboard', writer=self.writer)
        self.viz_start, self.viz_freq = viz_start, viz_freq

    def forward(self, descriptors, points, scores, score_map, depth_map, poses, K, imgs, env):
        def batch_project(pts):
            return self.projector.cartesian(pts, depth_map, poses, K)

        H, W = score_map.size(2), score_map.size(3)
        cornerness = self.beta[0] * self.score_corner(score_map, imgs, batch_project)
        distinction = self.beta[1] * self.desc_dist(descriptors)
        proj_pts, invis_idx = batch_project(points)
        match = self.beta[2] * self.desc_match(descriptors, points.unsqueeze(0), proj_pts, invis_idx, H, W)
        retention = self.beta[3] * self.desc_ret(points, depth_map, poses, K, descriptors, env[0])
        loss = distinction + cornerness + match + retention

        if self.writer is not None:
            self.writer.add_scalars('Loss', {'distinction': distinction,
                                             'cornerness': cornerness,
                                             'match': match,
                                             'retention': retention,
                                             'all': loss}, self.n_iter)

        if self.n_iter >= self.viz_start and self.n_iter % self.viz_freq == 0:
            self.viz.show(imgs, points, 'hot', values=scores.squeeze(-1).detach().cpu().numpy(), name='train', step=self.n_iter)

            self.viz.show(score_map, color='hot', vmax=0.01, name='score', step=self.n_iter)

            pair = torch.tensor([[0, 1], [0, 3], [0, 5], [0, 7]])
            b_src, b_dst = pair[:, 0], pair[:, 1]
            matched, confidence = ConsecutiveMatch()(descriptors[b_src], descriptors[b_dst], points[b_dst])
            top_conf, top_idx = confidence.topk(50, dim=1)
            top_conf, top_idx = top_conf.detach().cpu().numpy(), top_idx.unsqueeze(-1).repeat(1, 1, 2)
            self.viz.showmatch(imgs[b_src], points[b_src].gather(1, top_idx), imgs[b_dst], matched.gather(1, top_idx), 'hot', top_conf, 0.9, 1, name='match', step=self.n_iter)

        self.n_iter += 1
        return loss


class ScoreLoss(nn.Module):
    def __init__(self, radius=8, num_corners=500, debug=False):
        super(ScoreLoss, self).__init__()
        self.bceloss = nn.BCELoss()
        self.corner_det = kf.CornerGFTT()
        self.num_corners = num_corners
        self.pool = nn.MaxPool2d(kernel_size=radius, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=radius)
        self.debug = Visualizer(default_name='corners') if debug else debug

    def forward(self, scores_dense, imgs, projector):
        corners = self.get_corners(imgs, projector)
        lap = kn.filters.laplacian(scores_dense, 5) # smoothness

        if self.debug:
            _B = corners.shape[0]
            _coords = corners.squeeze().nonzero(as_tuple=False)
            _pts_list = [_coords[_coords[:, 0] == i][:, [2, 1]] for i in range(_B)]
            _pts = torch.ones(_B, max([p.shape[0] for p in _pts_list]), 2) * -2
            for i, p in enumerate(_pts_list):
                _pts[i, :len(p)] = p
            _pts = C.normalize_pixel_coordinates(_pts, imgs.shape[2], imgs.shape[3])
            self.debug.show(imgs, _pts)

        return self.bceloss(scores_dense, corners) + (scores_dense * torch.exp(-lap)).mean() * 10

    def get_corners(self, imgs, projector=None):
        (B, _, H, W), N = imgs.shape, self.num_corners
        corners = kf.nms2d(self.corner_det(kn.rgb_to_grayscale(imgs)), (5, 5))

        # only one in patch
        output, indices = self.pool(corners)
        corners = self.unpool(output, indices)

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


class DiscriptorDistinctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.cosine = PairwiseCosine(inter_batch=False)

    def forward(self, descriptors):
        pcos = self.cosine(descriptors, descriptors)
        return self.relu(pcos).mean()


class DiscriptorRententionLoss(nn.Module):
    def __init__(self, writer=None):
        super().__init__()
        self.writer = writer
        self.memory = {}
        self.projector = Projector()
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.n_iter = 0

    def forward(self, points, depth_map, pose, K, descriptors, env):
        if env not in self.memory:
            self.memory[env] = Memory(N=50000).to(points)
        memory = self.memory[env]

        points_w = self.projector.pix2world(points, depth_map, pose, K)
        mask = points_w.isfinite().all(dim=-1)
        valid_pts, valid_desc = points_w[mask], descriptors[mask]

        matched_desc, n_match = memory.write(valid_pts, valid_desc, match_count=True)

        if self.writer is not None:
            self.n_iter += 1
            self.writer.add_scalars('Misc/DiscriptorRentention', {
                'n_match': n_match
            }, self.n_iter)

        return 1 - self.cosine(valid_desc, matched_desc).mean()


class DiscriptorMatchLoss(nn.Module):
    def __init__(self, radius=1, writer=None):
        super(DiscriptorMatchLoss, self).__init__()
        self.radius, self.writer, self.n_iter = radius, writer, 0
        self.cosine = PairwiseCosine(inter_batch=True)

    def forward(self, descriptors, pts_src, pts_dst, invis_idx, height, width):
        pts_src = C.denormalize_pixel_coordinates(pts_src.detach(), height, width)
        pts_dst = C.denormalize_pixel_coordinates(pts_dst.detach(), height, width)

        dist = torch.cdist(pts_dst, pts_src)
        dist[tuple(invis_idx)] = float('nan')
        pcos = self.cosine(descriptors, descriptors)

        match = (dist <= self.radius).triu(diagonal=1)
        miss = (dist > self.radius).triu(diagonal=1)

        if self.writer is not None:
            self.n_iter += 1
            self.writer.add_scalars('Misc/DiscriptorMatch', {
                'n_match': match.sum(),
                'n_miss': miss.sum(),
            }, self.n_iter)

        return (1 - pcos[match].mean()) + pcos[miss].mean()


class PairwiseCosine(nn.Module):
    def __init__(self, inter_batch=False, dim=-1, eps=1e-8):
        super(PairwiseCosine, self).__init__()
        self.inter_batch, self.dim, self.eps = inter_batch, dim, eps
        self.eqn = 'amd,bnd->abmn' if inter_batch else 'bmd,bnd->bmn'

    def forward(self, x, y):
        xx = torch.sum(x**2, dim=self.dim).unsqueeze(-1) # (A, M, 1)
        yy = torch.sum(y**2, dim=self.dim).unsqueeze(-2) # (B, 1, N)
        if self.inter_batch:
            xx, yy = xx.unsqueeze(1), yy.unsqueeze(0) # (A, 1, M, 1), (1, B, 1, N)
        xy = torch.einsum(self.eqn, x, y)
        return xy / (xx * yy).clamp(min=self.eps**2).sqrt()
