#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
from collections import deque
import torch.nn as nn


def rectify_savepath(path, overwrite=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_path, save_file_dup = path, 0
    while os.path.exists(save_path) and not overwrite:
        save_file_dup += 1
        save_path = path + '.%d' % save_file_dup

    return save_path


def save_model(model, path):
    model = model.module if isinstance(model, nn.DataParallel) else model

    save_path = rectify_savepath(path)

    torch.save(model.state_dict(), save_path)
    print('Saved model: %s' % save_path)


def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    torch.save(model.state_dict(), path)
    print('Loaded model: %s' % path)
    return model


class GlobalStepCounter():
    def __init__(self, initial_step=0):
        self._steps = initial_step

    @property
    def steps(self):
        return self._steps

    def step(self, step=1):
        self._steps += 1
        return self._steps


class ProgressBarDescription():
    def __init__(self, tq, ave_steps=50):
        self.losses = deque()
        self.tq = tq
        self.ave_steps = ave_steps
    
    def update(self, loss):
        loss = loss.item()
        if np.isnan(loss):
            print('Warning: nan loss.')
        else:
            self.losses.append(loss)
            if len(self.losses) > self.ave_steps:
                self.losses.popleft()
        self.tq.set_description("Loss: %.4f at" % (np.average(self.losses)))


class Timer:
    def __init__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        torch.cuda.synchronize()
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.duration = time.time()-self.start_time
        self.start()
        return self.duration


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=0, verbose=False):
        super().__init__(optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=verbose)
        self.no_decrease = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return self._reduce_lr(epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
                return False
            else:
                return True
