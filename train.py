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
import torch.utils.data as Data

from datasets import TartanAir
from models import FeatureNet 
from models.utils import count_parameters, Timer



def test(loader, net, device):
    net.eval()
    with torch.no_grad():
        for batch_idx, (image) in enumerate(tqdm.tqdm(loader)):
            image = image.to(args.device)
            points, scores, features = net(image)
            # evaluation script
    return 0.9 # accuracy


def train(loader, net, device, creterion):
    net.train()
    for batch_idx, (image, depth, pose) in enumerate(tqdm.tqdm(loader)):
        image, depth, pose = image.to(args.device), depth.to(args.device), pose.to(args.device)
        points, scores, features = net(image)
        # loss and evaluation script
    return 0.9 # accuracy


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epoch", type=int, default=10, help="number of epoches")
    parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
    parser.add_argument("--iteration", type=int, default=5, help="number of training iteration")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_data = TartanAir(root=args.data_root, train=True)
    train_loader = Data.DataLoader(train_data, args.batch_size, True)
    test_data = TartanAir(root=args.data_root, train=False)
    test_loader = Data.DataLoader(test_data, args.batch_size, False)

    net = FeatureNet().to(args.device) if args.load is None else torch.load(args.load, args.device)

    creterion = nn.CrossEntropyLoss()

    timer = Timer()
    for epoch in range(args.epoch):
        train_acc = train(train_loader, net, args.device, creterion)

        if args.save is not None:
            torch.save(net, args.save)

    test_acc = test(test_loader, net, args.device)
    print("Train Acc: %.3f, Test Acc: %.3f, Timing: %.2fs"%(train_acc, test_acc, timer.end()))
