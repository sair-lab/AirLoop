#!/usr/bin/env python3

from torch.utils.data import DataLoader

from .tartanair import TartanAir
from .tartanair import AirSampler
from .tartanair import TartanAirTest
from .tartanair import AirAugment


def get_datasets(args):
    if args.dataset == 'tartanair':
        tartanair = TartanAir(args.train_root, args.scale, augment=False, catalog_path=args.train_catalog)
        train_data, eval_data = tartanair.rand_split(
            [1 - args.eval_percentage, args.eval_percentage], args.eval_split_seed)
        train_data.include_exclude(args.include, args.exclude)
        eval_data.include_exclude(args.include, args.exclude)

        shuffle = 'none'
        if 'envshuffle' in args.task:
            shuffle = 'env'
        elif 'seqshuffle' in args.task:
            shuffle = 'seq'
        elif 'allshuffle' in args.task:
            shuffle = 'all'

        train_sampler = AirSampler(train_data, args.batch_size, shuffle=shuffle, overlap=False)
        eval_sampler = AirSampler(eval_data, args.batch_size, shuffle='none', overlap=False)

        return make_loader(train_data, train_sampler, args), make_loader(eval_data, eval_sampler, args)


def make_loader(dataset, sampler, args):
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=True, num_workers=args.num_workers)
