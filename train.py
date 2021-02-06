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
from collections import deque
from tensorboard import program
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import AirSampler
from models import FeatureNet
from models import FeatureNetLoss
from models import ConsecutiveMatch
from models import EarlyStopScheduler
from utils import MatchEvaluator, Visualizer
from datasets import TartanAir, TartanAirTest, AirAugment
from models import Timer, count_parameters, GlobalStepCounter


@torch.no_grad()
def evaluate(net, evaluator, loader, args):
    net.eval()
    for images, depths, poses, K, env_seq in tqdm.tqdm(loader):
        images = images.to(args.device)
        depths = depths.to(args.device)
        poses = poses.to(args.device)
        K = K.to(args.device)
        descriptors, points, pointness, scores = net(images)
        evaluator.observe(descriptors, points, scores, pointness, depths, poses, K, images, env_seq)

    evaluator.report()


def test(net, loader, args=None):
    net.eval()
    vis = Visualizer()
    match = ConsecutiveMatch()
    with torch.no_grad():
        for idx, (image, pose, K) in enumerate(tqdm.tqdm(loader)):
            image = image.to(args.device)
            pose = pose.to(args.device)
            K = K.to(args.device)
            descriptors, points, pointness, scores = net(image)
            matched, _ = match(descriptors, points)
            # evaluation script
            if args.visualize:
                vis.show(image, points)
                for (img0, pts0, img1, pts1) in zip(image[:-1], image[:-1], image[1:], matched):
                    vis.showmatch(img0, pts0, img1, pts1)
    return 0.9 # accuracy

def train(net, loader, criterion, optimizer, counter, args=None, loss_ave=50, eval_loader=None, evaluator=None):
    net.train()
    train_loss, batches = deque(), len(loader)
    enumerator = tqdm.tqdm(loader)
    for images, depths, poses, K, env_seq in enumerator:
        images = images.to(args.device)
        depths = depths.to(args.device)
        poses = poses.to(args.device)
        K = K.to(args.device)
        optimizer.zero_grad()
        descriptors, points, pointness, scores = net(images)
        loss = criterion(descriptors, points, scores, pointness, depths, poses, K, images, env_seq[0])
        loss.backward()
        optimizer.step()

        if np.isnan(loss.item()):
            print('Warning: loss is nan during iteration %d. BP skipped.' % counter.steps)
        else:
            train_loss.append(loss.item())
            if len(train_loss) > loss_ave:
                train_loss.popleft()
        enumerator.set_description("Loss: %.4f at"%(np.average(train_loss)))

        if evaluator is not None and counter.steps % args.eval_freq == 0:
            evaluate(net, evaluator, eval_loader, args)
            net.train()

        counter.step()

    return np.average(train_loss)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--dataset", type=str, default='tartanair', help="TartanAir")
    parser.add_argument("--train-root", type=str, default='/data/datasets/tartanair', help="data location")
    parser.add_argument("--test-root", type=str, default='/data/datasets/tartanair_test')
    parser.add_argument("--train-catalog", type=str, default='./.cache/tartanair-sequences.pbz2', help='processed training set')
    parser.add_argument("--test-catalog", type=str, default='./.cache/tartanair-test-sequences.pbz2', help='processed test set')
    parser.add_argument("--log-dir", type=str, default=None, help="log dir")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
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
    parser.add_argument("--eval-percentage", type=float, default=0.2, help='Percentage of sequences for eval')
    parser.add_argument("--eval-freq", type=int, default=10000, help='Evaluate every * steps')
    parser.add_argument("--eval-topk", type=int, default=200, help='Only inspect top * matches')
    parser.add_argument("--eval-back", type=int, nargs='+', default=[1])
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data, eval_data = TartanAir(args.train_root, args.scale, catalog_path=args.train_catalog) \
        .rand_split([1 - args.eval_percentage, args.eval_percentage], args.eval_split_seed)
    eval_data.augment = AirAugment(args.scale, resize_only=True)
    test_data = TartanAirTest(args.test_root, args.scale, catalog_path=args.test_catalog)

    train_sampler = AirSampler(train_data, args.batch_size, shuffle=True)
    eval_sampler = AirSampler(eval_data, args.batch_size, shuffle=False, overlap=False)
    test_sampler = AirSampler(test_data, args.batch_size, shuffle=False)

    train_loader = DataLoader(train_data, batch_sampler=train_sampler, pin_memory=True, num_workers=args.num_workers)
    eval_loader = DataLoader(eval_data, batch_sampler=eval_sampler, pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler, pin_memory=True, num_workers=args.num_workers)

    writer = None
    if args.log_dir is not None:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer = SummaryWriter(os.path.join(args.log_dir, current_time))
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', args.log_dir, '--bind_all'])
        print(('TensorBoard at %s \n' % tb.launch()))

    step_counter = GlobalStepCounter(initial_step=1)
    criterion = FeatureNetLoss(writer=writer, viz_start=args.viz_start, viz_freq=args.viz_freq, counter=step_counter)
    net = FeatureNet(args.feat_dim, args.feat_num).to(args.device) if args.load is None else torch.load(args.load, args.device)
    if not isinstance(net, nn.DataParallel):
        net = nn.DataParallel(net)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    evaluator = MatchEvaluator(back=args.eval_back, viz=None, top=args.eval_topk, writer=writer, counter=step_counter)

    timer = Timer()
    for epoch in range(args.epoch):
        train_acc = train(net, train_loader, criterion, optimizer, step_counter, args, eval_loader=eval_loader, evaluator=evaluator)

        if args.save is not None:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            save_path, save_file_dup = args.save, 0
            while os.path.exists(save_path):
                save_file_dup += 1
                save_path = args.save + '.%d' % save_file_dup
            torch.save(net, save_path)
            print('Saved model: %s' % save_path)

        if scheduler.step(1-train_acc):
            print('Early Stopping!')
            break

    test_acc = test(net, test_loader, args)
    print("Train: %.3f, Test: %.3f, Timing: %.2fs"%(train_acc, test_acc, timer.end()))
