#!/usr/bin/env python3

import copy
import os
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
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

        if args.ll_weight_dir is not None:
            self.weight_dir = Path(args.ll_weight_dir)
            self.load(args.ll_weight_load, args.devices[0])
        else:
            self.weight_dir = None
        self.args = args

    def init_loss(self, model: nn.Module) -> None:
        '''Called once upon first `__call__`.'''

    def calc_loss(self, *args, **kwargs) -> torch.Tensor:
        '''Called with arguments from `__call__`.'''

    def restore_states(self, state: List) -> List[int]:
        '''Called with loaded states.'''
        return list(range(len(state)))

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

    def load(self, tasks, device):
        if self.weight_dir is None or tasks is None:
            return
        tasks = [tasks] if isinstance(tasks, str) else tasks
        paths = [str(self.weight_dir / f'{self.name.lower()}.{task}') for task in tasks]
        loaded_idx = self.restore_states([torch.load(path, map_location=device) for path in paths])
        print(f'Loaded {self.name} weights: {os.pathsep.join(np.take(paths, loaded_idx))}')

    def save(self, task=None, overwrite=True):
        if self.weight_dir is None:
            return
        path = str(self.weight_dir / f'{self.name.lower()}' if task is None else self.weight_dir / f'{self.name.lower()}.{task}')
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

    def restore_states(self, state: List) -> List[int]:
        self.old_imp = [torch.stack(ws).mean(dim=0) for ws in zip(*state)]
        return list(range(len(state)))

    def get_states(self) -> Any:
        return self.cur_imp

class MASLoss(L2RegLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100, relational=True):
        self.relational = relational
        name = 'RMAS' if relational else 'MAS'
        super().__init__(name, args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb)
        self.cosine = PairwiseCosine()

    def get_importance(self, model, gd):
        if self.relational:
            pcos = self.cosine(gd[None], gd[None])[0]
            norm = pcos.square().sum().sqrt()
        else:
            norm = gd.square().sum(dim=1).sqrt().mean()

        gs = ag.grad(norm, model.parameters(), retain_graph=True)
        return [g.abs() for g in gs], len(gd)


class EWCLoss(L2RegLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100, ce=False):
        self.ce = ce
        post_backward = not ce
        super().__init__('CEWC' if ce else 'EWC', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb, post_backward=post_backward)
        self.cosine = PairwiseCosine()

    def get_importance(self, *args, **kwargs):
        if self.ce:
            return self.get_importance_ce(*args, **kwargs)
        else:
            return self.get_importance_grad(*args, **kwargs)

    def get_importance_grad(self, model, gd):
        b = len(gd)
        assert b % 3 == 0
        gs = [p.grad for p in self.cur_param]

        return [g ** 2 for g in gs], b // 3 * 2

    def get_importance_ce(self, model, gd):
        b = len(gd)
        assert b % 3 == 0

        ank, pos, neg = torch.split(gd, b // 3)
        logit_p = F.softplus(F.cosine_similarity(ank, pos), beta=5, threshold=4.5).clamp(min=0, max=1)
        logit_n = F.softplus(F.cosine_similarity(ank, neg), beta=5, threshold=4.5).clamp(min=0, max=1)

        loss = (F.binary_cross_entropy(logit_p, torch.ones_like(logit_p)) +
                F.binary_cross_entropy(logit_n, torch.zeros_like(logit_n))) / 2

        gs = ag.grad(loss, model.parameters(), retain_graph=True)
        return [g ** 2 for g in gs], b // 3 * 2


class SILoss(L2RegLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100):
        super().__init__('SI', args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb, post_backward=True, avg_method='none')
        self.cosine = PairwiseCosine()
        self.last_param = None
        self.w = None
        self.eps = 1e-1

    def init_loss_sub(self) -> None:
        self.last_param = [w.data.clone() for w in self.cur_param]
        self.w = [torch.zeros_like(p) for p in self.cur_param]

    @torch.no_grad()
    def get_importance(self, model, gd):
        gs = [p.grad for p in self.cur_param]

        # path integral
        cur_param = [p.data.clone() for p in self.cur_param]
        for w, g, cur_p, last_p in zip(self.w, gs, cur_param, self.last_param):
            w -= g * (cur_p - last_p)
        self.last_param = cur_param

        omega = [pt - p0 for pt, p0 in zip(cur_param, self.old_param)]
        return [w / (omg ** 2 + self.eps) for w, omg in zip(self.w, omega)], len(gd)


class KDLoss(LifelongLoss):
    def __init__(self, args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=100, relational=True, last_only=True):
        self._model_t_states: List[nn.Module.T_destination] = []
        self.last_only = last_only
        self.relational = relational
        name = 'RKD' if relational else 'KD'
        name = ('' if last_only else 'C') + name
        super().__init__(name, args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb, post_backward=False)
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
        gd_s = gd
        loss = 0
        # distill from each teacher
        for model_t in self.model_t:
            with torch.no_grad():
                gd_t = model_t(img=img)

            if self.relational:
                response_s = self.cosine(gd_s[None], gd_s[None])[0]
                response_t = self.cosine(gd_t[None], gd_t[None])[0]
            else:
                response_s = gd_s
                response_t = gd_t

            loss += F.smooth_l1_loss(response_s, response_t) / len(self.model_t)

        return loss

    def restore_states(self, state: List) -> List[int]:
        self._model_t_states = state.copy()
        if self.last_only:
            self._model_t_states = self._model_t_states[-1:]
            return [len(state) - 1]
        else:
            return list(range(len(state)))

    def get_states(self) -> nn.Module.T_destination:
        module = self.model_s.module if isinstance(self.model_s, nn.DataParallel) else self.model_s
        return module.state_dict()


class CompoundLifelongLoss():
    def __init__(self, *losses: LifelongLoss):
        self.losses = losses

    def __call__(self, *args, model: nn.Module = None, **kwargs):
        return sum(loss(*args, model, **kwargs) for loss in self.losses)

    def load(self, tasks, device):
        for loss in self.losses:
            loss.load(tasks, device)

    def save(self, task=None, overwrite=True):
        for loss in self.losses:
            loss.save(task, overwrite)

    def __iter__(self):
        return iter(self.losses)
