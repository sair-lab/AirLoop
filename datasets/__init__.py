#!/usr/bin/env python3

from torch.utils.data import DataLoader

from .base import DefaultSampler
from .tartanair import TartanAir
from .nordland import Nordland
from .robotcar import RobotCar


def get_dataset(args):
    if args.dataset == 'tartanair':
        tartanair = TartanAir(args.dataset_root, args.scale, catalog_dir=args.catalog_dir)
        train_data, eval_data = tartanair.rand_split(
            [1 - args.eval_percentage, args.eval_percentage], args.eval_split_seed)
        if 'train' in args.task:
            data = train_data
        elif 'eval' in args.task:
            data = eval_data
    else:
        if args.dataset == 'nordland':
            dataset_cls = Nordland
        elif args.dataset == 'robotcar':
            dataset_cls = RobotCar
        else:
            raise ValueError(f'Unrecognized dataset: {args.dataset}')

        split = 'train' if 'train' in args.task else 'eval'
        data = dataset_cls(args.dataset_root, args.scale, split=split, catalog_dir=args.catalog_dir)

    seq_merge, env_merge = 'cat', 'cat'
    if 'joint' in args.task:
        env_merge = 'rand_interleave'

    data.include_exclude(args.include, args.exclude)
    sampler = DefaultSampler(data, args.batch_size, seq_merge=seq_merge, env_merge=env_merge, overlap=False)
    loader = DataLoader(data, batch_sampler=sampler, pin_memory=True, num_workers=args.num_workers)

    return loader
