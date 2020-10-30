#!/usr/bin/env python3

import os
import sys
import tqdm
import copy
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms as T

from utils import Visualization
from datasets import TartanAir
from models import FeatureNet
from models import FeatureNetLoss
from models import EarlyStopScheduler
from models import Timer, count_parameters


def test(net, loader, args=None):
    net.eval()
    vis = Visualization('test')
    with torch.no_grad():
        for idx, (image) in enumerate(tqdm.tqdm(loader)):
            image = image.to(args.device)
            descriptors, points, scores = net(image)
            # evaluation script
            if args.visualize:
                vis.show(image, points)
    return 0.9 # accuracy


def train(net, loader, criterion, optimizer, args=None):
    net.train()
    vis = Visualization('train', args.debug)
    for idx, (images, depths, poses) in enumerate(tqdm.tqdm(loader)):
        images, depths, poses = images.to(args.device), depths.to(args.device), poses.to(args.device)
        descriptors, points, pointness = net(images)

        # TEMP
        fx = 320.0 / 2
        fy = 320.0 / 2
        cx = 320.0 / 2
        cy = 240.0 / 2

        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]]).to(images)

        loss = criterion(descriptors, points, pointness, depths, poses, K, images)
        loss.backward()
        # TEMP

        # loss and evaluation script
        if args.visualize:
            vis.show(images, points)
    return 0.9 # accuracy


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--dataset", type=str, default='tartanair', help="TartanAir")
    parser.add_argument("--data-root", type=str, default='data/office2 tiny', help="data location")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--feat-dim", type=int, default=512, help="feature dimension")
    parser.add_argument("--feat-num", type=int, default=500, help="feature number")
    parser.add_argument('--resize', nargs='+', type=int, default=[240,320], help='image resize')
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--min-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="factor of lr")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of optim")
    parser.add_argument("--w-decay", type=float, default=0, help="weight decay of optim")
    parser.add_argument("--epoch", type=int, default=15, help="number of epoches")
    parser.add_argument("--batch-size", type=int, default=5, help="minibatch size")
    parser.add_argument("--patience", type=int, default=5, help="training patience")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(visualize=True, debug=True)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    image_transform = T.Compose([T.Resize(args.resize), T.ToTensor()])
    depth_transform = T.Compose([T.ToPILImage(mode='F'), image_transform])

    train_data = TartanAir(root=args.data_root, train=True, img_transform=image_transform, depth_transform=depth_transform)
    train_loader = Data.DataLoader(train_data, args.batch_size, False)
    test_data = TartanAir(root=args.data_root, train=False, img_transform=image_transform)
    test_loader = Data.DataLoader(test_data, args.batch_size, False)

    criterion = FeatureNetLoss()
    net = FeatureNet(args.feat_dim, args.feat_num).to(args.device) if args.load is None else torch.load(args.load, args.device)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    timer = Timer()
    for epoch in range(args.epoch):
        train_acc = train(net, train_loader, criterion, optimizer, args)

        if args.save is not None:
            torch.save(net, args.save)

        if scheduler.step(1-train_acc):
            print('Early Stopping!')
            break

    test_acc = test(net, test_loader, args)
    print("Train: %.3f, Test: %.3f, Timing: %.2fs"%(train_acc, test_acc, timer.end()))
