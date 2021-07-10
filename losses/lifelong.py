#!/usr/bin/env python3

import copy
import os
from typing import List, Tuple

import torch
import torch.autograd as ag
import torch.nn.functional as F

from utils import PairwiseCosine
from utils.misc import rectify_savepath


class WeightRegularizationLoss():
    def __init__(self, name, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
        self.writer, self.counter = writer, counter
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq
        self.weights = None
        self.old_weights = None
        self.old_params = None
        self.name = name
        self.n_observed = 0

        if args.ll_weights_load is not None:
            self.load(args.ll_weights_load)
        self.save_path = args.ll_weights_save

    def get_weights(self, model, *args, **kwargs) -> Tuple[List[torch.Tensor], int]:
        raise NotImplementedError()

    def __call__(self, model, *args, **kwargs):
        if self.old_weights is None:
            self.old_weights = [torch.zeros_like(t.data) for t in model.parameters()]
        if self.old_params is None:
            self.weights = [torch.zeros_like(t.data) for t in model.parameters()]
            self.old_params = [t.data.clone() for t in model.parameters()]
        if self.old_weights[0].device != self.old_params[0].device:
            self.old_weights = [ow.to(self.old_params[0].device) for ow in self.old_weights]

        gs, bs = self.get_weights(model, *args, **kwargs)
        loss = 0
        for g, param, old_param, ow, w in zip(gs, model.parameters(), self.old_params, self.old_weights, self.weights):
            w.data = (w * self.n_observed + g) / (self.n_observed + bs)
            loss += (ow * (param - old_param) ** 2).sum()
        self.n_observed += bs

        return loss

    def load(self, paths):
        paths = [paths] if isinstance(paths, str) else paths
        weights = [torch.load(path) for path in paths]
        self.old_weights = [torch.stack(ws).mean(dim=0) for ws in zip(*weights)]
        print(f'Loaded {self.name} weights: {os.pathsep.join(paths)}')

    def save(self, path=None, task=None, overwrite=True):
        path = self.save_path if path is None else path
        if path is None:
            return
        path = path if task is None else f'{path}.{task}'
        save_path = rectify_savepath(path, overwrite=overwrite)
        torch.save(self.weights, save_path)
        print(f'Saved {self.name} weights: {save_path}')


class MASLoss(WeightRegularizationLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
        super().__init__('MAS', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter)
        self.cosine = PairwiseCosine()

    def get_weights(self, model, gd):
        (gd, _) = gd
        pcos = self.cosine(gd[None], gd[None])[0]
        norm = pcos.square().sum().sqrt()

        gs = ag.grad(norm, model.parameters(), retain_graph=True)
        return [g.abs() for g in gs], len(gd)


class EWCLoss(WeightRegularizationLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
        super().__init__('EWC', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter)

    def get_weights(self, model, gd):
        (gd, _) = gd
        b = len(gd)
        assert b % 3 == 0

        ank, pos, neg = torch.split(gd, b // 3)
        logit_p = F.softplus(F.cosine_similarity(ank, pos), beta=5, threshold=4.5).clamp(min=0, max=1)
        logit_n = F.softplus(F.cosine_similarity(ank, neg), beta=5, threshold=4.5).clamp(min=0, max=1)

        loss = (F.binary_cross_entropy(logit_p, torch.ones_like(logit_p)) +
                F.binary_cross_entropy(logit_n, torch.zeros_like(logit_n))) / 2

        gs = ag.grad(loss, model.parameters(), retain_graph=True)
        return [g ** 2 for g in gs], b // 3 * 2


class RKDLoss():
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
        self.writer, self.counter = writer, counter
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq
        self.cosine = PairwiseCosine()
        self.teacher_model = None

    def __call__(self, model, gd, img):
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(model)
            self.teacher_model.eval()

        gd_s, _ = gd
        with torch.no_grad():
            gd_t = self.teacher_model(img=img)

        pcos_s = self.cosine(gd_s[None], gd_s[None])[0]
        pcos_t = self.cosine(gd_t[None], gd_t[None])[0]

        return F.smooth_l1_loss(pcos_s, pcos_t)

    def save(self, path=None, task=None, overwrite=True):
        pass

    def load(self, paths):
        pass
