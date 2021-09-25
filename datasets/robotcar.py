#!/usr/bin/env python3

import os
import pathlib

from PIL import Image
import numpy as np
import pandas as pd
from scipy import interpolate
import torch
from torchvision.transforms import functional as F

from .augment import AirAugment
from .base import DatasetBase

class RobotCar(DatasetBase):
    WEATHER_TAGS = ['sun', 'overcast', 'night']

    def __init__(self, root, scale=1, split='train', catalog_dir=None):
        self.gps = {}
        self.img_ts = {}
        # ! tmp
        self.split = split
        super().__init__(pathlib.Path(root) / 'robotcar' / split, f'robotcar-{split}', catalog_dir)

        self.augment = AirAugment(scale, size=[480, 640], resize_only=True)
        self.trimmed_size = [480, 640]

    def _populate(self):
        seqs = self.path_prefix.glob('*-*-*-*-*-*')
        for seq in seqs:
            tags = open(seq / 'tags.csv', 'r').readline().strip().split(',')
            env = [t for t in tags if t in self.WEATHER_TAGS]
            if len(env) == 1:
                self.seqs.setdefault(env[0], []).append(seq.name)

        self.img_gps, self.img_ts, self.vel, self.heading = {}, {}, {}, {}
        for env, seq in self.get_env_seqs():
            seq_path = self.path_prefix / seq

            # gps = np.loadtxt(seq_path / 'gps/gps.csv', delimiter=',', skiprows=1, usecols=(0, 8, 9),
            gps = np.loadtxt(seq_path / 'gps/ins.csv', delimiter=',', skiprows=1, usecols=(0, 5, 6, 9, 10, 11, 14),
                             dtype=[('ts', 'i8'), ('loc', '2f8'), ('vel', '3f8'), ('yaw', 'f8')])
            gps = np.sort(gps, order=['ts'])
            # deduplicate timestamp
            selected = np.ones(len(gps), dtype=bool)
            selected[1:] = gps['ts'][1:] != gps['ts'][:-1]
            gps = gps[selected]

            img_ts = np.array([int(p.split('.')[-2])
                               for p in os.listdir(seq_path / 'stereo/centre') if p.endswith('.png')])
            img_ts = np.sort(img_ts)

            # prevent precision cutoff
            offset = gps['ts'][0]
            gps_interp = interpolate.interp1d(gps['ts'] - offset, gps['loc'], axis=0, bounds_error=False,
                                              fill_value=np.nan, assume_sorted=True)
            img_gps = gps_interp((img_ts - offset))

            valid = np.isfinite(img_gps).all(axis=1)
            if valid.sum() >= 1000:
                # ! tmp
                if self.split == 'test':
                    valid[min(len(valid), 12000):] = False
                self.img_gps[env, seq], self.img_ts[env, seq] = img_gps[valid], img_ts[valid].astype(np.int64)
                # self.vel[env, seq] = np.sqrt(np.sum(gps['vel'] ** 2, 1))
                self.heading[env, seq] = gps['yaw']
            else:
                self.seqs[env].remove(seq)
                if len(self.seqs[env]) == 0:
                    self.seqs.pop(env)

        return ['img_gps', 'img_ts', 'heading']

    def get_size(self, env, seq):
        return len(self.img_ts[env, seq])

    def getitem_impl(self, env, seq, idx):
        try:
            img_path = self.path_prefix / seq / f'stereo/centre/{self.img_ts[env, seq][idx]}.png'
            arr = np.array(Image.open(img_path)) / 255.0
            g0, r, b, g1 = arr[0::2, 0::2], arr[1::2, 0::2], arr[0::2, 1::2], arr[1::2, 1::2]
            g = (g0 + g1) / 2
            image = torch.from_numpy(np.stack([r, g, b]).astype(np.float32))
            image = F.center_crop(image, self.trimmed_size)
        except Exception as e:
            print('Bad image: %s:%s:%d: %s' % (env, seq, idx, str(e)))
            image = Image.new('RGB', self.trimmed_size)
        image = self.augment(image)[0]
        aux = (self.img_gps[env, seq][idx], np.rad2deg(self.heading[env, seq][idx]))
        return image, aux

    def summary(self):
        return pd.DataFrame(data=[
            [env, seq, self.get_size(env, seq)] for env, seq in self.get_env_seqs()], 
            columns=['env', 'seq', 'size'])

