#!/usr/bin/env python3

import torch
import torch.nn as nn


class PoseNet(nn.Module):
    """Predicts pose from pairs of images.

    See: SfMLearner.
    """

    conv_k, conv_s, conv_p = 3, 2, 1
    conv_ch = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, n_target=1):
        """
        Args:
          n_target: Number of 3-channel target images to compare with input[0:2]
                    for pose.
        """
        super(PoseNet, self).__init__()

        self.n_target = n_target

        # 7 down conv, 1 1x1 conv, ave-pool
        num_chan = [(n_target + 1) * 3] + self.conv_ch
        layers = []
        for in_ch, out_ch in zip(num_chan[:-1], num_chan[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch,
                                    kernel_size=self.conv_k, stride=self.conv_s,
                                    padding=self.conv_p))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(num_chan[-1], 6 * n_target, kernel_size=1))
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.pose = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
          x: 1 source and multiple target images staked along channel dimension
            (N, 3 * (n_target + 1), H, W).

        Returns:
          Relative pose of each target w.r.t. the first image (N, n_target, 6).
        """
        pose = self.pose(x)
        return pose.reshape(len(x), self.n_target, 6)


if __name__ == "__main__":
    '''Test code'''
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Test FeatureNet')
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cuda:0, or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--batch-size', type=int, default=30, help='number of minibatch size')
    parser.add_argument('--crop-size', nargs='+', type=int, default=[320, 320], help='image crop size')
    parser.add_argument('--n-target', type=int, default=2, help='# of target images')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = PoseNet(args.n_target).to(args.device)
    inputs = torch.randn(args.batch_size, (args.n_target + 1)
                         * 3, *args.crop_size).to(args.device)

    start = time.time()
    with torch.no_grad():
        for i in range(5):
            pose = net(inputs)
            torch.cuda.empty_cache()
            print(pose.shape)
    print('time:', time.time() - start)
