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


def load_model(model, path, device='cuda', strict=False):
    #! tmp
    state_dict = torch.load(path, map_location=device)
    state_dict_ = {}
    for k, v in state_dict.items():
        k: str = k[25:] if k.startswith('features.encoder.encoder') else k
        state_dict_[k] = v
    state_dict = state_dict_
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if len(missing_keys) > 0:
            print(f'Warning: Missing key(s): {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'Warning: Unexpected key(s): {unexpected_keys}')
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
        self.hist = []
        self.start_time = None
        self.n_iter = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_trace):
        torch.cuda.synchronize()
        self.hist.append(time.time() - self.start_time)
        self.start_time = None

    def get_ave(self):
        return np.average(self.hist)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
