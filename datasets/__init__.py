#!/usr/bin/env python3

from torch.utils.data import DataLoader

from .base import DefaultSampler
from .tartanair import TartanAir, AirAugment
from .nordland import Nordland
from .robotcar import RobotCar


def get_datasets(args):
    if args.dataset == 'tartanair':
        tartanair = TartanAir(args.dataset_root, args.scale, augment=False, catalog_dir=args.catalog_dir)
        pretrain_data, train_data, eval_data = tartanair.rand_split(
            [args.pretrain_percentage,
             1 - args.pretrain_percentage - args.eval_percentage,
             args.eval_percentage], args.eval_split_seed)

        if 'pretrain' in args.task:
            data = pretrain_data
        elif 'train' in args.task:
            data = train_data
        elif 'eval' in args.task:
            data = eval_data
    elif args.dataset == 'nordland':
        if 'pretrain' in args.task or 'train' in args.task:
            nordland = Nordland(args.dataset_root, args.scale, augment=False, split='train', catalog_dir=args.catalog_dir)
            pretrain_data, train_data = nordland.rand_split(
                [args.pretrain_percentage,
                 1 - args.pretrain_percentage], args.eval_split_seed)
            data = pretrain_data if 'pretrain' in args.task else train_data
        elif 'eval' in args.task:
            data = Nordland(args.dataset_root, args.scale, augment=False, split='test', catalog_dir=args.catalog_dir)
    elif args.dataset == 'robotcar':
        if 'pretrain' in args.task or 'train' in args.task:
            data = RobotCar(args.dataset_root, args.scale, augment=False, split='train', catalog_dir=args.catalog_dir)
        elif 'eval' in args.task:
            data = RobotCar(args.dataset_root, args.scale, augment=False, split='test', catalog_dir=args.catalog_dir)

    seq_merge, env_merge = 'cat', 'cat'
    if 'joint' in args.task:
        seq_merge, env_merge = 'cat', 'rand_pick'

    data.include_exclude(args.include, args.exclude)
    sampler = DefaultSampler(data, args.batch_size, seq_merge=seq_merge, env_merge=env_merge, overlap=False)
    loader = DataLoader(data, batch_sampler=sampler, pin_memory=True, num_workers=args.num_workers)
    res = data.augment.img_size

    return loader, res
