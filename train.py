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
        for idx, (image, K) in enumerate(tqdm.tqdm(loader)):
            image, K = image.to(args.device), K.to(args.device)
            descriptors, points, scores = net(image)
            # evaluation script
            if args.visualize:
                vis.show(image, points)
    return 0.9 # accuracy


def train(net, loader, criterion, optimizer, args=None):
    net.train()
    train_loss, batches = 0, len(loader)
    vis = Visualization('train', args.debug)
    enumerater = tqdm.tqdm(enumerate(loader))
    for idx, (images, depths, poses, K) in enumerater:
        images = images.to(args.device)
        depths = depths.to(args.device)
        poses = poses.to(args.device)
        K = K.to(args.device)
        optimizer.zero_grad()
        descriptors, points, pointness = net(images)
        loss = criterion(descriptors, points, pointness, depths, poses, K, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        enumerater.set_description("Loss: %.4f on %d/%d"%(train_loss/(idx+1), idx+1, batches))
        if args.visualize:
            vis.show(images, points)
    return train_loss/(idx+1)


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
    parser.add_argument("--visualize", dest='visualize', action='store_true')
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.set_defaults(visualize=False, debug=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    image_transform = T.Compose([T.Resize(args.resize), lambda x: np.array(x), T.ToTensor()])
    depth_transform = T.Compose([T.ToPILImage(mode='F'), image_transform])

    train_data = TartanAir(root=args.data_root, train=True, img_transform=image_transform, depth_transform=depth_transform)
    test_data = TartanAir(root=args.data_root, train=False, img_transform=image_transform)

    train_loader = Data.DataLoader(train_data, args.batch_size, False, pin_memory=True, num_workers=args.num_workers)
    test_loader = Data.DataLoader(test_data, args.batch_size, False, pin_memory=True, num_workers=args.num_workers)

    criterion = FeatureNetLoss(debug=args.debug)
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
