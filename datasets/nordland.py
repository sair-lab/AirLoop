#!/usr/bin/env python3

import pathlib

from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F

from .augment import AirAugment
from .base import DatasetBase

class Nordland(DatasetBase):

    def __init__(self, root, scale=1, augment=True, split='train', catalog_dir=None):
        self.split = split
        super().__init__(pathlib.Path(root) / 'nordland', f'nordland-{split}', catalog_dir)

        self.augment = AirAugment(scale, size=[480, 640], resize_only=not augment)
        self.trimmed_size = [1080, 1440]

    def _populate(self):
        seqs = ['section1', 'section2'] if self.split == 'train' else ['section1', 'section2', 'section3']
        self.seqs.update({env: seqs.copy() for env in ['spring', 'summer', 'fall', 'winter']})

        self.seq_lims = {}
        for env_seq in self.get_env_seqs():
            indices = [int(f.stem) for f in self._get_paths(*env_seq, '*')]
            self.seq_lims[env_seq] = (min(indices), max(indices) + 1)

        return ['seq_lims']

    def get_size(self, env, seq):
        lims = self.seq_lims[env, seq]
        return lims[1] - lims[0]

    def getitem_impl(self, env, seq, idx):
        offset = self.seq_lims[env, seq][0] + idx
        try:
            image = F.center_crop(Image.open(list(self._get_paths(env, seq, offset))[0]), self.trimmed_size)
        except Exception as e:
            print('Bad image: %s:%s:%d: %s' % (env, seq, idx, str(e)))
            image = Image.new('RGB', self.trimmed_size)
        image = self.augment(image)[0]
        return image, offset

    def summary(self):
        return pd.DataFrame(data=[
            [env, seq, self.get_size(env, seq)] for env, seq in self.get_env_seqs()], 
            columns=['env', 'seq', 'size'])

    def _get_paths(self, env, seq, idx):
        return (self.path_prefix / self.split / ('%s_images_%s' % (env, self.split)) / seq).glob(str(idx) + '.png')
