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

from models import FeatureNet 
from datasets import TartanAir
from models.utils import Timer
from models.utils import count_parameters
from models.utils import EarlyStopScheduler


def test(net, loader, device):
    net.eval()
    with torch.no_grad():
        for batch_idx, (image) in enumerate(tqdm.tqdm(loader)):
            image = image.to(args.device)
            points, scores, features = net(image)
            # evaluation script
    return 0.9 # accuracy


def train(net, loader, device, creterion, optimizer):
    net.train()
    for batch_idx, (image, depth, pose) in enumerate(tqdm.tqdm(loader)):
        image, depth, pose = image.to(args.device), depth.to(args.device), pose.to(args.device)
        points, scores, features = net(image)
        # loss and evaluation script
    return 0.9 # accuracy


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--dataset", type=str, default='tartanair', help="TartanAir")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="data location")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--min-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="factor of lr")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of optim")
    parser.add_argument("--w-decay", type=float, default=0, help="weight decay of optim")
    parser.add_argument("--epoch", type=int, default=15, help="number of epoches")
    parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
    parser.add_argument("--patience", type=int, default=5, help="training patience")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_data = TartanAir(root=args.data_root, train=True)
    train_loader = Data.DataLoader(train_data, args.batch_size, True)
    test_data = TartanAir(root=args.data_root, train=False)
    test_loader = Data.DataLoader(test_data, args.batch_size, False)

    criterion = nn.CrossEntropyLoss()
    net = FeatureNet().to(args.device) if args.load is None else torch.load(args.load, args.device)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    timer = Timer()
    for epoch in range(args.epoch):
        train_acc = train(net, train_loader, args.device, criterion, optimizer)

        if args.save is not None:
            torch.save(net, args.save)

        if scheduler.step(1-train_acc):
            print('Early Stopping!')
            break

    test_acc = test(net, test_loader, args.device)
    print("Train: %.3f, Test: %.3f, Timing: %.2fs"%(train_acc, test_acc, timer.end()))
