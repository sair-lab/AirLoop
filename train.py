#!/usr/bin/env python3

import os
import sys
import tqdm
import copy
import torch
import random
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboard import program
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from models import FeatureNet
from datasets import get_datasets
from models.loss import MemReplayLoss
from utils.evaluation import MatchEvaluator, RecognitionEvaluator
from utils.misc import save_model, load_model, GlobalStepCounter, ProgressBarDescription


@torch.no_grad()
def evaluate(net, loader, counter, args, writer=None):
    net.eval()

    evaluators = []

    if 'match' in args.task:
        evaluators.append(MatchEvaluator(back=args.eval_back, viz=None, top=args.eval_topk, writer=writer, counter=counter, args=args))
    if 'recog' in args.task:
        evaluators.append(RecognitionEvaluator(loader=loader, n_feature=args.feat_dim, D_frame=args.gd_dim, args=args))

    for images, depths, poses, K, env_seq in tqdm.tqdm(loader):
        images = images.to(args.device)
        depths = depths.to(args.device)
        poses = poses.to(args.device)
        K = K.to(args.device)
        descriptors, points, pointness, scores, gd, _ = net(images)
        for evaluator in evaluators:
            evaluator.observe(descriptors, points, scores, gd, pointness, depths, poses, K, images, env_seq)

    for evaluator in evaluators:
        evaluator.report()


def train(model, loader, optimizer, counter, args, writer=None):
    model.train()

    if 'train' in args.task:
        criterion = MemReplayLoss(writer=writer, viz_start=args.viz_start, viz_freq=args.viz_freq, counter=counter, args=args)

    last_env = None

    for epoch in range(args.epoch):
        enumerator = tqdm.tqdm(loader)
        pbd = ProgressBarDescription(enumerator)
        for images, depths, poses, K, env_seq in enumerator:
            images = images.to(args.device)
            depths = depths.to(args.device)
            poses = poses.to(args.device)
            K = K.to(args.device)

            loss = criterion(model, images, depths, poses, K, env_seq[0])

            # in case loss is manually set to 0 to skip batches
            if loss.requires_grad and not loss.isnan():
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # save model on env change for env-incremental tasks
            if 'env' in args.task and last_env != env_seq[0][0]:
                if last_env is not None:
                    save_model(model, '%s.%s' % (args.save, last_env))
                last_env = env_seq[0][0]
            
            if counter.steps % args.save_freq == 0:
                save_model(model, '%s.step%d' % (args.save, counter.steps))

            pbd.update(loss)
            counter.step()

        if 'env' in args.task:
            if args.save is not None:
                save_model(model, '%s.%s' % (args.save, last_env))
            if args.mem_save is not None:
                criterion.memory.save('%s.%s' % (args.mem_save, last_env))
        else:
            save_model(model, '%s.epoch%d' % (args.save, epoch))


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    loader, img_res = get_datasets(args)
    if args.devices is None:
        args.devices = ['cuda:%d' % i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu']
    args.device = args.devices[0]

    model = FeatureNet(img_res, args.feat_dim, args.feat_num, args.gd_dim).to(args.device)
    if args.load:
        load_model(model, args.load, device=args.device)
    if not args.no_parallel:
        model = nn.DataParallel(model, device_ids=args.devices)

    writer = None
    if args.log_dir is not None:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer = SummaryWriter(os.path.join(args.log_dir, current_time))
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', args.log_dir, '--bind_all', '--samples_per_plugin=images=50'])
        print(('TensorBoard at %s \n' % tb.launch()))

    step_counter = GlobalStepCounter(initial_step=1)

    if 'train' in args.task:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        train(model, loader, optimizer, step_counter, args, writer)
    if 'eval' in args.task:
        evaluate(model, loader, step_counter, args, writer)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--task", type=str, choices=['pretrain', 'train-envseq', 'train-envshuffle', 'train-seqshuffle', 'train-allshuffle', 'eval-recog', 'eval-match', 'eval-match-recog'], default='train-envseq')
    parser.add_argument("--no-parallel", action='store_true', help="DataParallel")
    parser.add_argument("--devices", type=str, nargs='+', default=None, help="Available devices")
    parser.add_argument("--dataset", type=str, default='tartanair', help="TartanAir")
    parser.add_argument("--include", type=str, default=None, help="sequences to include")
    parser.add_argument("--exclude", type=str, default=None, help="sequences to exclude")
    parser.add_argument("--train-root", type=str, default='/data/datasets/tartanair', help="data location")
    parser.add_argument("--test-root", type=str, default='/data/datasets/tartanair_test')
    parser.add_argument("--train-catalog", type=str, default='./.cache/tartanair-sequences.pbz2', help='processed training set')
    parser.add_argument("--test-catalog", type=str, default='./.cache/tartanair-test-sequences.pbz2', help='processed test set')
    parser.add_argument("--log-dir", type=str, default=None, help="log dir")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model")
    parser.add_argument("--mem-load", type=str, default=None, help="load saved memory")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--save-freq", type=int, default=33558, help="model saving frequency")
    parser.add_argument("--mem-save", type=str, default=None, help="memory save path")
    parser.add_argument("--mas", action='store_true', help="enable MAS")
    parser.add_argument("--gd-dim", type=int, default=1024, help="global descriptor dimension")
    parser.add_argument("--feat-dim", type=int, default=256, help="feature dimension")
    parser.add_argument("--feat-num", type=int, default=500, help="feature number")
    parser.add_argument('--scale', type=float, default=1, help='image resize')
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="factor of lr")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of optim")
    parser.add_argument("--w-decay", type=float, default=0, help="weight decay of optim")
    parser.add_argument("--epoch", type=int, default=15, help="number of epoches")
    parser.add_argument("--batch-size", type=int, default=8, help="minibatch size")
    parser.add_argument("--patience", type=int, default=5, help="training patience")
    parser.add_argument("--num-workers", type=int, default=4, help="workers of dataloader")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--viz_start", type=int, default=np.inf, help='Visualize starting from iteration')
    parser.add_argument("--viz_freq", type=int, default=1, help='Visualize every * iteration(s)')
    parser.add_argument("--eval-split-seed", type=int, default=42, help='Seed for splitting the dataset')
    parser.add_argument("--pretrain-percentage", type=float, default=0.0, help='Percentage of sequences for eval')
    parser.add_argument("--eval-percentage", type=float, default=0.2, help='Percentage of sequences for eval')
    parser.add_argument("--eval-freq", type=int, default=10000, help='Evaluate every * steps')
    parser.add_argument("--eval-topk", type=int, default=200, help='Only inspect top * matches')
    parser.add_argument("--eval-back", type=int, nargs='+', default=[1])
    parser.add_argument("--eval-save", type=str, help='Evaluation save path')
    parserd_args = parser.parse_args(); print(parserd_args)

    main(parserd_args)
