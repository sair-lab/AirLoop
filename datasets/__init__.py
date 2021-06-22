#!/usr/bin/env python3

from torch.utils.data import DataLoader

from .base import DefaultSampler
from .tartanair import TartanAir
from .tartanair import AirAugment


def get_datasets(args):
    if args.dataset == 'tartanair':
        tartanair = TartanAir(args.train_root, args.scale, augment=False, catalog_dir=args.catalog_dir)
        res = tartanair.augment.img_size
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

        return make_loader(data, sampler, args), res


def make_loader(dataset, sampler, args):
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=True, num_workers=args.num_workers)
