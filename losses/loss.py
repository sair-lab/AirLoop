#!/usr/bin/env python3

import numpy as np
import torch

from utils import Visualizer
from models.memory import TartanAirMemory, NordlandMemory, RobotCarMemory
from utils import Projector, PairwiseCosine, gen_probe

from .lifelong import get_ll_loss


class MemReplayLoss():
    def __init__(self, writer=None, viz_start=float('inf'), viz_freq=200, counter=None, args=None):
        super().__init__()
        self.args = args
        self.writer, self.counter, self.viz_start, self.viz_freq = writer, counter, viz_start, viz_freq
        self.viz = Visualizer('tensorboard', writer=self.writer)

        if args.dataset == 'tartanair':
            self.memory = TartanAirMemory(capacity=args.mem_size, n_probe=1200, swap_dir=args.mem_swap, out_device=args.device)
        elif args.dataset == 'nordland':
            self.memory = NordlandMemory(capacity=args.mem_size, swap_dir=args.mem_swap, out_device=args.device)
        elif args.dataset == 'robotcar':
            self.memory = RobotCarMemory(capacity=args.mem_size, dist_tol=20, head_tol=15, swap_dir=args.mem_swap, out_device=args.device)
        if args.mem_load is not None:
            self.memory.load(args.mem_load)

        self.n_triplet, self.n_recent, self.n_pair = 4, 0, 1
        self.min_sample_size = 32

        self.gd_match = GlobalDescMatchLoss(n_triplet=self.n_triplet, n_pair=self.n_pair, writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)
        self.ll_loss = get_ll_loss(args, writer=writer, viz=self.viz, viz_start=viz_start, viz_freq=viz_freq, counter=self.counter)

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

        loss = 0
        gd = net(img=img)
        gd_match = self.gd_match(gd)
        loss += gd_match

        # forgetting prevention
        if self.ll_loss is not None:
            for ll_loss in self.ll_loss:
                loss_name = ll_loss.name.lower()
                if loss_name in ['mas', 'rmas', 'ewc', 'cewc', 'si']:
                    loss += ll_loss(model=net, gd=gd)
                elif loss_name in ['kd', 'rkd', 'ifgir']:
                    loss += ll_loss(model=net, gd=gd, img=img)
                else:
                    raise ValueError(f'Unrecognized lifelong loss: {ll_loss}')

        # logging and visualization
        if self.writer is not None:
            n_iter = self.counter.steps if self.counter is not None else 0
            self.writer.add_scalars('Loss', {'global': gd_match}, n_iter)
            self.writer.add_histogram('Misc/RelN', neg_rel, n_iter)
            self.writer.add_histogram('Misc/RelP', pos_rel, n_iter)
            self.writer.add_scalars('Misc/MemoryUsage', {'len': len(self.memory)}, n_iter)

            # show triplets
            if self.viz is not None and n_iter >= self.viz_start and n_iter % self.viz_freq == 0:
                H, W = img.shape[2:]
                if isinstance(self.memory, TartanAirMemory):
                    N = ank_batch['pos'].shape[1]

                    # project points from pos to ank
                    mem_pts_scr = Projector.world2pix(pos_batch['pos'].reshape(-1, N, 3), (H, W), ank_batch['pose'], ank_batch['K'], ank_batch['depth_map'])[0]
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
        if isinstance(self.memory, TartanAirMemory):
            depth_map, pose, K = aux
            points_w = Projector.pix2world(gen_probe(depth_map), depth_map, pose, K)
            self.memory.store_fifo(pos=points_w, img=imgs, depth_map=depth_map, pose=pose, K=K)
        elif isinstance(self.memory, NordlandMemory):
            offset = aux
            self.memory.store_fifo(img=imgs, offset=offset)
        elif isinstance(self.memory, RobotCarMemory):
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


class GlobalDescMatchLoss():

    def __init__(self, n_triplet=8, n_pair=1, writer=None,
                 viz=None, viz_start=float('inf'), viz_freq=200, counter=None, debug=False):
        super().__init__()
        self.counter, self.writer = counter, writer
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq
        self.cosine = PairwiseCosine()
        self.n_triplet, self.n_pair = n_triplet, n_pair

    def __call__(self, gd):
        gd_a, gd_p, gd_n = gd.split([self.n_triplet, self.n_triplet * self.n_pair, self.n_triplet * self.n_pair])
        gd_a = gd_a[:, None]
        gd_p = gd_p.reshape(self.n_triplet, self.n_pair, *gd_p.shape[1:])
        gd_n = gd_n.reshape(self.n_triplet, self.n_pair, *gd_n.shape[1:])

        sim_ap = self.cosine(gd_a, gd_p)
        sim_an = self.cosine(gd_a, gd_n)
        triplet_loss = (sim_an - sim_ap + 1).clamp(min=0)

        # logging
        n_iter = self.counter.steps
        if self.writer is not None:
            self.writer.add_histogram('Misc/RelevanceLoss', triplet_loss, n_iter)
            self.writer.add_histogram('Misc/SimAP', sim_ap, n_iter)
            self.writer.add_histogram('Misc/SimAN', sim_an, n_iter)
            self.writer.add_scalars('Misc/GD', {'2-Norm': torch.norm(gd, dim=-1).mean()}, n_iter)

        return triplet_loss.mean()
