#!/usr/bin/env python3

import time
import torch


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.duration = time.time()-self.start_time
        self.start()
        return self.duration


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                    verbose=False, threshold=1e-4, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
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


def pix2world(p, depth, T_p, K_inv):
    """Transforms from pixel coordinates to world frame.

    Args:
      p:     Pixel coordinates (N, 2).
      depth: Depth of each point (N).
      T_p:   Camera poses in which p is observed (N, 3, 4).
      K_inv: Inverse of camera intrinsics (N, 3, 3).

    Returns:
      Coordinates in world frame (N, 3).
    """
    N = len(p)

    p_h = torch.cat([p, torch.ones(N, 1, device=p.get_device())], 1).unsqueeze(2)
    p_cam = torch.bmm(K_inv, p_h) * depth.reshape(N, 1, 1)
    # T_p^-1 * p_cam
    R, t = T_p[:, :, :3], T_p[:, :, 3].unsqueeze(2)
    p_world_h = torch.bmm(R.transpose(1, 2), p_cam - t).squeeze(2)
    return p_world_h


def world2pix(p, T_p, K):
    """Transforms world frame to pixel coordinates.

    Args:
      p:   World coordinates (N, 3).
      T_p: Camera poses in which p is observed (N, 3, 4).
      K:   Camera intrinsics (N, 3, 3).

    Returns:
      Pixel corrdinates (N, 2).
    """
    N = len(p)

    p_h = torch.cat([p, torch.ones(N, 1, device=p.get_device())], 1).unsqueeze(2)
    p_cam_h = torch.bmm(T_p, p_h)
    pix_h = torch.bmm(K, p_cam_h).squeeze(2)
    return pix_h[:, :2] / pix_h[:, 2].unsqueeze(1)


def project_points(p, depth, T_p, T_q, K, K_inv=None):
    """Projects p visible in pose T_p to pose T_q.

    Args:
      p:     List of points (N, 2).
      depth: Depth of each point(N).
      T_p:   List of camera poses in which p is observed (N, 3, 4).
      T_q:   List of camera poses to project into (N, 3, 4).
      K:     Camera intrinsics (3, 3).
      K_inv: Optional; precomputed inverse of K (3, 3).

    Returns:
      Coordinates of p in pose T_q (N, 2).
    """
    world_coord = pix2world(p, depth, T_p, K_inv if K_inv else torch.inverse(K))
    return world2pix(world_coord, T_q, K)
