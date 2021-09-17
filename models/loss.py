#!/usr/bin/env python3

import kornia as kn
import kornia.feature as kf
import kornia.geometry.conversions as C
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import AirAugment
from utils import Visualizer
from models.memory import SIoUMemory, OffsetMemory, LocationMemory
from utils import GridSample, Projector, PairwiseCosine, ConsecutiveMatch, gen_probe
from losses import get_ll_loss


class MemReplayLoss():
    def __init__(self, beta=[1, 1, 1, 1], writer=None, viz_start=float('inf'), viz_freq=200, counter=None, args=None):
        super().__init__()
        self.writer, self.beta, self.counter = writer, beta, counter
        self.viz_start, self.viz_freq = viz_start, viz_freq
        self.viz = Visualizer() if self.writer is None else Visualizer('tensorboard', writer=self.writer)
        self.projector = Projector()
        self.grid_sample = GridSample()
        self.augment = None
        if args.dataset == 'tartanair':
            self.memory = SIoUMemory(capacity=args.mem_size, n_probe=1200, swap_dir='.cache/memory', out_device='cpu' if self.augment is not None else args.device)
        elif args.dataset == 'nordland':
            self.memory = OffsetMemory(capacity=args.mem_size, swap_dir='.cache/memory', out_device='cpu' if self.augment is not None else args.device)
        elif args.dataset == 'robotcar':
            self.memory = LocationMemory(capacity=args.mem_size, dist_tol=20, head_tol=15, swap_dir='.cache/memory', out_device='cpu' if self.augment is not None else args.device)
        if args.mem_load is not None:
            self.memory.load(args.mem_load)
        self.score_corner = ScoreLoss(writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        self.n_triplet, self.n_recent, self.n_pair = 4, 0, 1
        self.min_sample_size = 32
        self.gd_match = GlobalDescMatchLoss(n_triplet=self.n_triplet, n_pair=self.n_pair, writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        if args.gd_only:
            self.score_corner, self.desc_match = None, None
        else:
            self.score_corner = ScoreLoss(writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
            self.desc_match = DiscriptorMatchLoss(writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        self.ll_loss = get_ll_loss(args, writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        self.args = args

    def __call__(self, net, img, aux, env):
        device = img.device
        self.store_memory(img, aux, env)

        if len(self.memory) < self.min_sample_size:
            return torch.zeros(1).to(device)

        _, (ank_batch, pos_batch, neg_batch), (pos_rel, neg_rel) = \
            self.memory.sample_frames(self.n_triplet, self.n_recent, self.n_pair)

        # no suitable triplet
        if ank_batch is None:
            return torch.zeros(1).to(device)

        img = recombine('img', ank_batch, pos_batch, neg_batch)

        # only kind of memory with full groundtruth
        if isinstance(self.memory, SIoUMemory):
            depth_map = recombine('depth_map', ank_batch, pos_batch, neg_batch)
            pose = recombine('pose', ank_batch, pos_batch, neg_batch)
            K = recombine('K', ank_batch, pos_batch, neg_batch)
            if self.augment is not None:
                augmented = [(self.augment(img_, K_, depth_map_)) for img_, K_, depth_map_ in zip(img, K, depth_map)]
                img, K, depth_map = [torch.stack(list(tensor)).to(device) for tensor in zip(*augmented)]
                pose = pose.to(device)
        elif self.augment is not None:
            img = torch.stack([self.augment(img_).to(device) for img_ in img])

        loss = 0
        if self.args.gd_only:
            gd = net(img=img)
        else:
            descriptors, points, score_map, scores, gd = net(img=img)
            def batch_project(pts):
                return self.projector.cartesian(pts, depth_map, pose, K)

            def sim_metric(x, y):
                cosine = PairwiseCosine(inter_batch=True)
                return cosine(x, y)
                # return net(desc0=x, desc1=y)
            cornerness = self.beta[0] * self.score_corner(score_map, img, batch_project)
            neg_st = self.n_triplet * (1 + self.n_pair)
            desc_match = self.beta[1] * self.desc_match(sim_metric, img[:neg_st], points[:neg_st], descriptors[:neg_st], scores[:neg_st], depth_map[:neg_st], pose[:neg_st], K[:neg_st])
            loss += cornerness + desc_match
        gd_match = self.beta[2] * self.gd_match(gd)
        loss += gd_match

        # forgetting prevention
        if self.ll_loss is not None:
            if self.args.ll_method.lower() in ['mas', 'ewc', 'si']:
                loss += self.ll_loss(model=net, gd=gd)
            elif self.args.ll_method.lower() == 'rkd':
                loss += self.ll_loss(model=net, gd=gd, img=img)
            else:
                raise ValueError(f'Unrecognized lifelong loss: {self.ll_loss}')

        if self.writer is not None:
            n_iter = self.counter.steps if self.counter is not None else 0
            self.writer.add_scalars('Loss', {'global': gd_match}, n_iter)
            if not self.args.gd_only:
                self.writer.add_scalars('Loss', {'cornerness': cornerness,
                                                 'desc': desc_match,
                                                 'all': loss}, n_iter)
            self.writer.add_histogram('Misc/RelN', neg_rel, n_iter)
            self.writer.add_histogram('Misc/RelP', pos_rel, n_iter)
            self.writer.add_scalars('Misc/MemoryUsage', {'len': len(self.memory)}, n_iter)

            # show triplets
            if self.viz is not None and n_iter >= self.viz_start and n_iter % self.viz_freq == 0:
                if not self.args.gd_only:
                    self.viz.show(img, points, 'hot', values=scores.squeeze(-1).detach().cpu().numpy(), name='Out/Points', step=n_iter)

                H, W = img.shape[2:]
                if isinstance(self.memory, SIoUMemory):
                    N = ank_batch['pos'].shape[1]

                    # project points from pos to ank
                    mem_pts_scr = self.projector.world2pix(pos_batch['pos'].reshape(-1, N, 3), (H, W), ank_batch['pose'], ank_batch['K'], ank_batch['depth_map'])[0]
                    B_total = self.n_triplet * (self.n_pair * 2 + 1)
                    mem_pts_scr_ = mem_pts_scr.reshape(self.n_triplet, self.n_pair * N, 2)
                    proj_pts_ = torch.cat([
                        torch.zeros_like(mem_pts_scr_).fill_(np.nan),
                        mem_pts_scr_,
                        torch.zeros_like(mem_pts_scr_).fill_(np.nan)], 1).reshape(B_total, self.n_pair * N, 2)

                    proj_pts_color = torch.arange(self.n_pair)[None, :, None].expand(B_total, self.n_pair, N) + 1
                    proj_pts_color = proj_pts_color.reshape(B_total, self.n_pair * N).detach().cpu().numpy()
                else:
                    proj_pts_ = proj_pts_color = None

                ank_img, pos_img, neg_img = img.split([self.n_triplet, self.n_triplet * self.n_pair, self.n_triplet * self.n_pair])
                self.viz.show(
                    torch.cat([pos_img.reshape(self.n_triplet, self.n_pair, 3, H, W), ank_img[:, None], neg_img.reshape(self.n_triplet, self.n_pair, 3, H, W)], dim=1).reshape(-1, 3, H, W),
                    proj_pts_, 'tab10', values=proj_pts_color, vmin=0, vmax=10, name='Misc/GlobalDesc/Triplet', step=n_iter,
                    nrow=(self.n_pair * 2 + 1))

        return loss

    def store_memory(self, imgs, aux, env):
        self.memory.swap(env[0])
        if isinstance(self.memory, SIoUMemory):
            depth_map, pose, K = aux
            points_w = self.projector.pix2world(gen_probe(depth_map), depth_map, pose, K)
            self.memory.store_fifo(pos=points_w, img=imgs, depth_map=depth_map, pose=pose, K=K)
        elif isinstance(self.memory, OffsetMemory):
            offset = aux
            self.memory.store_fifo(img=imgs, offset=offset)
        elif isinstance(self.memory, LocationMemory):
            location, heading = aux
            self.memory.store_fifo(img=imgs, location=location, heading=heading)


def recombine(key, *batches):
    tensors = [batch[key] for batch in batches]
    reversed_shapes = [list(reversed(tensor.shape[1:])) for tensor in tensors]
    common_shapes = []
    for shape in zip(*reversed_shapes):
        if all(s == shape[0] for s in shape):
            common_shapes.insert(0, shape[0])
        else:
            break
    return torch.cat([tensor.reshape(-1, *common_shapes) for tensor in tensors])


class ScoreLoss(nn.Module):
    def __init__(self, radius=8, num_corners=500, writer=None,
                 viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
        super(ScoreLoss, self).__init__()
        self.bceloss = nn.BCELoss()
        self.corner_det = kf.CornerGFTT()
        self.num_corners = num_corners
        self.pool = nn.MaxPool2d(kernel_size=radius, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=radius)
        self.counter = counter
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq

    def forward(self, scores_dense, imgs, projector):
        corners = self.get_corners(imgs, projector)
        corners = kn.filters.gaussian_blur2d(corners, kernel_size=(7, 7), sigma=(1, 1))
        lap = kn.filters.laplacian(scores_dense, 5) # smoothness

        n_iter = self.counter.steps
        if n_iter >= self.viz_start and n_iter % self.viz_freq == 0:
            self.viz.show(scores_dense, color='hot', name='Out/Score', step=n_iter)

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


class GlobalDescMatchLoss():
    eps = 1e-2
    thr = 0.3

    def __init__(self, n_triplet=8, n_pair=1, writer=None,
                 viz=None, viz_start=float('inf'), viz_freq=200, counter=None, debug=False):
        super().__init__()
        self.writer = writer
        self.projector = Projector()
        self.cosine = PairwiseCosine()
        self.counter = counter
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq * 10
        self.debug = debug
        self.hist = {'imgs': None, 'depth_map': None, 'pose': None, 'K': None, 'len': 0}
        self.imgs_hist, self.depth_map, self.pose, self.K = [None] * 4
        self.n_triplet, self.n_pair = n_triplet, n_pair

    def __call__(self, gd):
        gd, gd_locs = gd

        gd_a, gd_p, gd_n = gd.split([self.n_triplet, self.n_triplet * self.n_pair, self.n_triplet * self.n_pair])
        gd_a = gd_a[:, None]
        gd_p = gd_p.reshape(self.n_triplet, self.n_pair, *gd_p.shape[1:])
        gd_n = gd_n.reshape(self.n_triplet, self.n_pair, *gd_n.shape[1:])

        gd_norm = torch.norm(gd, dim=-1)
        # loss = self.triplet_loss(gd_a, gd_p, gd_n) + gd_norm * 1e-3
        sim_ap = self.cosine(gd_a, gd_p)
        sim_an = self.cosine(gd_a, gd_n)
        triplet = (sim_an - sim_ap + 1).clamp(min=0)
        # gd_loc_sp = torch.norm(F.normalize(gd_locs, dim=2), p=1, dim=2).mean(dim=1)
        # gd_den = torch.norm(F.normalize(gd_locs.sum(dim=1), dim=1), p=1, dim=1)
        loss = triplet.mean() # + (gd_loc_sp + F.relu(30 - gd_den)) * 1e-2

        n_iter = self.counter.steps
        if self.writer is not None:
            self.writer.add_histogram('Misc/RelevanceLoss', triplet, n_iter)
            self.writer.add_histogram('Misc/SimAP', sim_ap, n_iter)
            self.writer.add_histogram('Misc/SimAN', sim_an, n_iter)
            self.writer.add_scalars('Misc/GD', {
                # 'LocSparsity': gd_loc_sp.mean(),
                # 'GDDensity': gd_den.mean(),
                '2-Norm': gd_norm.mean()}, n_iter)

        return loss.mean()


class DiscriptorMatchLoss(nn.Module):
    eps = 1e-6

    def __init__(self, radius=1, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, debug=False):
        super(DiscriptorMatchLoss, self).__init__()
        self.radius, self.writer, self.counter = radius, writer, counter
        self.cosine = PairwiseCosine(inter_batch=True)
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq
        self.debug = debug
        self.projector = Projector()

    def forward(self, metric, images, points, descriptors, scores, depth_map, poses, K):
        height, width = images.shape[-2:]
        proj_pts, invis_idx = self.projector.cartesian(points, depth_map, poses, K)
        pts_src, pts_dst = points.unsqueeze(0), proj_pts

        pts_src = C.denormalize_pixel_coordinates(pts_src.detach(), height, width)
        pts_dst = C.denormalize_pixel_coordinates(pts_dst.detach(), height, width)

        dist = torch.cdist(pts_dst, pts_src)
        dist[tuple(invis_idx)] = float('nan')

        # !hardcode
        pair = torch.tensor([[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]])
        b_src, b_dst = torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])
        # *non-cartesian
        # dist = dist[b_src, b_dst]

        match = (dist <= self.radius).triu(diagonal=1)
        miss = (dist > self.radius).triu(diagonal=1)

        scores = scores.detach()
        # !hardcode
        score_ave = (scores[:, None, :, None] + scores[None, :, None, :]).clamp(min=self.eps) / 2
        pcos = metric(descriptors, descriptors)
        # *non-cartesian
        # score_ave = (scores[b_src, :, None] + scores[b_dst, None, :]).clamp(min=self.eps) / 2
        # pcos = metric(descriptors[b_src], descriptors[b_dst])

        sig_match = -torch.log(score_ave[match])
        sig_miss  = -torch.log(score_ave[miss])

        s_match = pcos[match]
        s_miss = pcos[miss]

        n_iter = self.counter.steps
        if self.writer is not None:
            self.writer.add_scalars('Misc/DiscriptorMatch/Count', {
                'n_match': match.sum(),
                'n_miss': miss.sum(),
            }, n_iter)

            if len(sig_match) > 0:
                self.writer.add_histogram('Misc/DiscriptorMatch/Sim/match', s_match, n_iter)
                self.writer.add_histogram('Misc/DiscriptorMatch/Sim/miss', s_miss[:len(s_match)], n_iter)

            if n_iter >= self.viz_start and n_iter % self.viz_freq == 0:
                rand_end = 0#-points.shape[1] // 2
                matched, confidence = ConsecutiveMatch()(descriptors[b_src, rand_end:], descriptors[b_dst, rand_end:], points[b_dst, rand_end:])
                top_conf, top_idx = confidence.topk(50, dim=1)
                top_conf, top_idx = top_conf.detach().cpu().numpy(), top_idx.unsqueeze(-1).repeat(1, 1, 2)
                self.viz.showmatch(images[b_src], points[b_src].gather(1, top_idx), images[b_dst], matched.gather(1, top_idx), 'hot', top_conf, 0.9, 1, name='Out/Match', step=n_iter)


        # match/mismatch blending factor
        f = lambda d: 0.75 - (d - self.radius) * (d + self.radius) / (4 * self.radius**2)

        alpha_match = f(dist[match].clamp(max=self.radius * 2))
        alpha_miss = f(dist[miss].clamp(max=self.radius * 2))

        loss_match, loss_miss = self.nll(sig_match, s_match, alpha_match), self.nll(sig_miss, s_miss, alpha_miss)

        loss_miss, loss_miss_idx = loss_miss.topk(match.sum() * 8)

        if self.debug and n_iter >= self.viz_start and n_iter % self.viz_freq == 0:
            B, *shape = images.shape
            src_images = images.unsqueeze(1).expand(B, B, *shape).reshape(B**2, *shape)
            dst_images = images.unsqueeze(0).expand(B, B, *shape).reshape(B**2, *shape)

            N = pts_src.shape[2]

            pts_src = pts_src.squeeze(0)
            src_pts_ = C.normalize_pixel_coordinates(pts_src, height, width)
            src_pts = torch.zeros(B**2, N * 2, 2).to(src_pts_).fill_(np.nan)
            dst_pts = torch.zeros(B**2, N * 2, 2).to(src_pts_).fill_(np.nan)
            vals = torch.zeros(B * B, N * 2).to(scores).fill_(np.nan)

            # match pairs
            match_b, match_src, match_dst = match.reshape(B**2, N, N).nonzero(as_tuple=True)
            src_pts[match_b, match_src] = src_pts_[match_b // B, match_src]
            dst_pts[match_b, match_src] = src_pts_[match_b % B, match_dst]
            vals[match_b, match_src] = 2 - loss_match

            # miss pairs
            miss_b, miss_src, miss_dst = miss.reshape(B**2, N, N).nonzero(as_tuple=True)
            miss_b, miss_src, miss_dst = miss_b[loss_miss_idx], miss_src[loss_miss_idx], miss_dst[loss_miss_idx]
            src_pts[miss_b, miss_src + N] = src_pts_[miss_b // B, miss_src]
            dst_pts[miss_b, miss_src + N] = src_pts_[miss_b % B, miss_dst]
            vals[miss_b, miss_src + N] = loss_miss - 2

            vrange = vals[vals.isfinite()].abs().max()
            self.viz.showmatch(src_images, src_pts, dst_images, dst_pts,
                'gnuplot2', vals.detach().cpu().numpy(), -vrange, vrange, name='desc', step=n_iter, nrow=B)

        return loss_match.mean() + loss_miss.mean()

    def nll(self, sig, cos, match=1, topk=None):
        # p(x) = exp(-l / sig) * C; l = 1 - x if match else x
        norm_const = torch.log(sig * (1 - torch.exp(-1 / sig)))
        loss = ((1 - cos) * match + cos * (1 - match)) / sig + norm_const
        return loss

