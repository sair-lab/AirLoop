#!/usr/bin/env python3

import copy
import os
from typing import Any, List, Tuple

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel

from utils import PairwiseCosine
from utils.misc import rectify_savepath


class LifelongLoss():
    def __init__(self, name, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=1, post_backward=False):
        self.writer, self.counter = writer, counter
        self.viz, self.viz_start, self.viz_freq = viz, viz_start, viz_freq
        self.name = name
        self.initialized = False
        self.post_backward = post_backward
        self.lamb = lamb

        self._call_args, self._call_kwargs = None, None

        if args.ll_weights_load is not None:
            self.load(args.ll_weights_load, args.devices[0])
        self.save_path = args.ll_weights_save

    def init_loss(self, model: nn.Module) -> None:
        '''Called once upon first `__call__`.'''

    def calc_loss(self, *args, **kwargs) -> torch.Tensor:
        '''Called with arguments from `__call__`.'''

    def restore_states(self, state: List) -> None:
        '''Called with loaded states.'''

    def get_states(self) -> Any:
        '''Called when saving.'''
        raise NotImplementedError()

    def _calc_log_loss(self, *args, **kwargs):
        '''Calculate loss and maybe log it.'''
        loss = self.lamb * self.calc_loss(*args, **kwargs)

        if self.writer is not None:
            n_iter = self.counter.steps if self.counter is not None else 0
            self.writer.add_scalars('Loss', {self.name.lower(): loss}, n_iter)

        return loss

    def __call__(self, *args, model: nn.Module = None, **kwargs):
        '''This loss will be called both before and after ``loss.backward()`` in case the method
        requires gradient information. Specifically, ``model = None`` if called as ```closure```
        after ``loss.backward()`` in ``optimizer.step(closure)``.'''

        if not self.initialized:
            self.init_loss(model)
            self.initialized = True

        if self.post_backward:
            if model is not None:
                # save context
                self._call_args, self._call_kwargs = args, kwargs
                return 0
            else:
                loss = self._calc_log_loss(*self._call_args, **self._call_kwargs)
                loss.backward()
                return loss
        elif model is not None:
            return self._calc_log_loss(*args, **kwargs)
        else:
            return 0

    def load(self, paths, device):
        paths = [paths] if isinstance(paths, str) else paths
        self.restore_states([torch.load(path, map_location=device) for path in paths])
        print(f'Loaded {self.name} weights: {os.pathsep.join(paths)}')

    def save(self, path=None, task=None, overwrite=True):
        path = self.save_path if path is None else path
        if path is None:
            return
        path = path if task is None else f'{path}.{task}'
        save_path = rectify_savepath(path, overwrite=overwrite)
        torch.save(self.get_states(), save_path)
        print(f'Saved {self.name} weights: {save_path}')


class L2RegLoss(LifelongLoss):
    def __init__(self, name, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=1, post_backward=False, avg_method='avg'):
        # used in super().load
        self.cur_imp, self.old_imp = None, None
        super().__init__(name, args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb, post_backward=post_backward)
        # buffers
        self.cur_param, self.old_param = None, None
        self.model = None
        self.n_observed = 0
        self.avg_method = avg_method

    def init_loss_sub(self) -> None:
        '''Loss specific initialization'''

    def get_importance(self, model: nn.Module, gd: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        '''Returns per-parameter importance as well as nominal batch size.'''
        raise NotImplementedError()

    def init_loss(self, model: nn.Module) -> None:
        assert model is not None
        self.model = model
        # save params from last task
        self.cur_param = list(model.parameters())
        self.old_param = [t.data.clone() for t in self.cur_param]
        # 0 importance if not set, ensure same device if loaded
        self.cur_imp = [torch.zeros_like(t.data) for t in self.cur_param]
        if self.old_imp is None:
            self.old_imp = [torch.zeros_like(t.data) for t in self.cur_param]
        elif self.old_imp[0].device != self.old_param[0].device:
            self.old_imp = [ow.to(self.old_param[0].device) for ow in self.old_imp]

        self.init_loss_sub()

    def calc_loss(self, *args, **kwargs) -> torch.Tensor:
        '''Collect weights for current task and penalize with those from previous tasks.'''
        gs, bs = self.get_importance(self.model, *args, **kwargs)

        loss = 0
        for imp, param, old_param, ow, w in zip(gs, self.cur_param, self.old_param, self.old_imp, self.cur_imp):
            if self.avg_method == 'avg':
                w.data = (w * self.n_observed + imp) / (self.n_observed + bs)
            elif self.avg_method == 'none':
                w.data = imp
            loss += (ow * (param - old_param) ** 2).sum()
        self.n_observed += bs

        return loss

    def restore_states(self, state: List) -> None:
        self.old_imp = [torch.stack(ws).mean(dim=0) for ws in zip(*state)]

    def get_states(self) -> Any:
        return self.cur_imp

class MASLoss(L2RegLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100):
        super().__init__('MAS', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb)
        self.cosine = PairwiseCosine()

    def get_importance(self, model, gd):
        (gd, _) = gd
        pcos = self.cosine(gd[None], gd[None])[0]
        norm = pcos.square().sum().sqrt()

        gs = ag.grad(norm, model.parameters(), retain_graph=True)
        return [g.abs() for g in gs], len(gd)


class EWCLoss(L2RegLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100):
        super().__init__('EWC', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb)

    def get_importance(self, model, gd):
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


class RKDLoss(LifelongLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100):
        self._model_t_states: List[nn.Module.T_destination] = []
        super().__init__('RKD', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb, post_backward=False)
        self.cosine = PairwiseCosine()
        self.model_s: nn.Module = None
        self.model_t: List[nn.Module] = []

    def init_loss(self, model: nn.Module) -> None:
        self.model_s = model
        self.model_t = []
        for model_t_state in self._model_t_states:
            model_t = copy.deepcopy(model).eval()
            (model_t.module if isinstance(model_t, nn.DataParallel) else model_t).load_state_dict(model_t_state)
            self.model_t.append(model_t)

    def calc_loss(self, gd: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        gd_s, _ = gd

        loss = 0
        # distill from each teacher
        for model_t in self.model_t:
            with torch.no_grad():
                gd_t = model_t(img=img)

            pcos_s = self.cosine(gd_s[None], gd_s[None])[0]
            pcos_t = self.cosine(gd_t[None], gd_t[None])[0]

            loss += F.smooth_l1_loss(pcos_s, pcos_t) / len(self.model_t)

        return loss

    def restore_states(self, state: List) -> None:
        self._model_t_states = state.copy()

    def get_states(self) -> nn.Module.T_destination:
        module = self.model_s.module if isinstance(self.model_s, nn.DataParallel) else self.model_s
        return module.state_dict()
