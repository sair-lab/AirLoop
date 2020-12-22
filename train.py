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
from torchvision import transforms as T
from torch.utils.data import DataLoader

from utils import Visualization
from datasets import AirSampler
from models import FeatureNet
from models import FeatureNetLoss
from models import ConsecutiveMatch
from models import EarlyStopScheduler
from models import Timer, count_parameters
from datasets import TartanAir, TartanAirTest


def test(net, loader, args=None):
    net.eval()
    vis = Visualization('test')
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


def train(net, loader, criterion, optimizer, args=None, loss_ave=50):
    net.train()
    train_loss, batches = deque([0] * loss_ave), len(loader)
    vis_train = Visualization('train', args.debug)
    vis_match = Visualization('match', args.debug)
    vis_score = Visualization('score', args.debug)
    match = ConsecutiveMatch()
    enumerater = tqdm.tqdm(enumerate(loader))
    for idx, (images, depths, poses, K) in enumerater:
        images = images.to(args.device)
        depths = depths.to(args.device)
        poses = poses.to(args.device)
        K = K.to(args.device)
        optimizer.zero_grad()
        descriptors, points, pointness, scores = net(images)
        loss = criterion(descriptors, points, pointness, depths, poses, K, images)
        loss.backward()
        optimizer.step()
        train_loss.popleft()
        train_loss.append(loss.item())
        if np.isnan(loss.item()):
            train_loss[-1] = np.mean(list(train_loss)[:-1])
            print('Warning: loss is nan during iteration %d.' % idx)
        enumerater.set_description("Loss: %.4f on %d/%d"%(sum(train_loss)/(loss_ave), idx+1, batches))
        if idx > args.visualize:
            vis_train.show(images, points, 'hot', values=scores.squeeze(-1).detach().cpu().numpy())

            vis_score.show(pointness, color='hot', vmax=0.01)

            matched, confidence = match(descriptors[[0, -1]], points[[0, -1]])
            for (img0, pts0, img1, pts1, conf) in zip(images[[0]], points[[0]], images[[-1]], matched, confidence):
                top_conf, top_idx = conf.topk(100)
                top_conf = top_conf.detach().cpu().numpy()
                vis_match.showmatch(img0, pts0[top_idx], img1, pts1[top_idx], 'hot', top_conf, 0.9, 1)

    return sum(train_loss)/(loss_ave)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--dataset", type=str, default='tartanair', help="TartanAir")
    parser.add_argument("--data-root", type=str, default='/data/datasets/tartanair', help="data location")
    parser.add_argument("--test-root", type=str, default='/data/datasets/tartanair_test')
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
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch size")
    parser.add_argument("--patience", type=int, default=5, help="training patience")
    parser.add_argument("--num-workers", type=int, default=4, help="workers of dataloader")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--visualize", type=int, nargs='?', default=np.inf, action='store', const=1000, help='Visualize starting from iteration')
    parser.add_argument("--debug", default=False, action='store_true')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data = TartanAir(args.data_root, args.scale)
    test_data = TartanAirTest(args.test_root, args.scale)

    train_sampler = AirSampler(train_data, args.batch_size, shuffle=True)
    test_sampler = AirSampler(test_data, args.batch_size, shuffle=False)

    train_loader = DataLoader(train_data, batch_sampler=train_sampler, pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler, pin_memory=True, num_workers=args.num_workers)

    criterion = FeatureNetLoss(debug=args.debug)
    net = FeatureNet(args.feat_dim, args.feat_num).to(args.device) if args.load is None else torch.load(args.load, args.device)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    timer = Timer()
    for epoch in range(args.epoch):
        train_acc = train(net, train_loader, criterion, optimizer, args)

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
