#!/usr/bin/env python3

from torch.utils.data import DataLoader

from .base import DefaultSampler
from .tartanair import TartanAir, AirAugment
from .nordland import Nordland


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

    shuffle = 'none'
    if 'envshuffle' in args.task:
        shuffle = 'env'
    elif 'seqshuffle' in args.task:
        shuffle = 'seq'
    elif 'allshuffle' in args.task or 'pretrain' in args.task:
        shuffle = 'all'
    elif 'eval' in args.task:
        shuffle = 'none'

    data.include_exclude(args.include, args.exclude)
    sampler = DefaultSampler(data, args.batch_size, shuffle=shuffle, overlap=False)
    loader = DataLoader(data, batch_sampler=sampler, pin_memory=True, num_workers=args.num_workers)
    res = data.augment.img_size

    return loader, res
