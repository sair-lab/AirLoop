import sys
import math
import torch
import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LevenbergMarquardt(Optimizer):
    '''
    Levenberg–Marquardt algorithm
    args:
    damping: damping factor λ (scalar)
    max_block: size of maximum block for matrix inversion
    (J^T J + λ I) * δ = J^T (y-f(x)) = -J^T loss
    https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
    Note: LM optim has no learning rate (lr), but has damping factor.
    This code re-uses the implementation for lr in PyTorch.
    Two options to save memory:
        1. Put parameters into several groups. (Manually, Suggested)
            See the code examples in nn.Modules.parameters()
        2. Set max_block as a large integer. (Automatically)
            Smaller max_block uses less memory but unstable optimization.
    For first choice, see the following site for per-parameter options.
    https://pytorch.org/docs/stable/optim.html
    '''
    def __init__(self, params, damping, max_block=sys.maxsize):
        # inheriting lr mechanism for damping factor
        assert damping > 0, 'Invalid Damping Factor.'
        self.loss, self.max_block = 0, max_block
        defaults = dict(lr=damping)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss, closure=None):
        self.loss += loss
        for group in self.param_groups:
            numels, D = [p.numel() for p in group['params'] if p.grad is not None], []
            Js = torch.cat([p.grad.data.view(1,-1) for p in group['params'] if p.grad is not None], -1)
            for J in Js.split(math.ceil(Js.size(1)/math.ceil(Js.size(1)/self.max_block)), dim=1):
                A = (J.T @ J) + group['lr'] * torch.eye(J.size(-1)).to(J)
                D.append(J.T.cholesky_solve(A.cholesky()))
            D = torch.cat(D).split(numels)
            [p.add_(-d.view(p.shape)*loss) for p,d in zip(group['params'], D) if p.grad is not None]


class UpDownDampingScheduler(_LRScheduler):
    '''
    UpDownDampingScheduler for LevenbergMarquardt Optimizer
    '''
    def __init__(self, optimizer, gamma, verbose=False):
        assert gamma > 1, 'Invalid Gamma.'
        self.gamma, self.loss = gamma, float('Inf')
        super().__init__(optimizer, verbose=verbose, last_epoch=-1)

    def get_lr(self):
        factor = 1/self.gamma if self.optimizer.loss < self.loss else self.gamma
        self.loss = self.optimizer.loss
        self.optimizer.loss = 0
        return [group['lr']*factor for group in self.optimizer.param_groups]


if __name__ == "__main__":
    '''
    Test Optimizer
    '''
    import argparse
    from torch import nn
    from tool import Timer
    import torch.utils.data as Data
    from torchvision.datasets import MNIST
    from torchvision import transforms as T

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--epoch', default=20, type=int, help='epoch')
    parser.add_argument('--batch-size', default=1000, type=int, help='epoch')
    parser.add_argument('--damping', default=2, type=float, help='Damping factor')
    parser.add_argument('--gamma', default=2, type=float, help='Gamma')
    args = parser.parse_args()

    # Easy Test
    class QuadNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(10, 10))

        def forward(self):
            return (self.w**2).sum()


    print('Testing Quadratic function without Scheduler...')
    net = QuadNet().to(args.device)
    optimizer = LevenbergMarquardt(net.parameters(), damping=args.damping)
    timer = Timer()
    for idx in range(50):
        optimizer.zero_grad()
        y = net()
        loss = y
        loss.backward()
        optimizer.step(loss)
        print('Quad loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()), end=' ')
        print('Using Damping factor', [group['lr'] for group in optimizer.param_groups])
        if loss < 1e-7:
            print('Early Stoping!')
            print('Optimization Early Done with loss:', loss.item())
            break


    print('Quadratic test with UpDownDampingScheduler...')
    net = QuadNet().to(args.device)
    optimizer = LevenbergMarquardt(net.parameters(), damping=args.damping)
    scheduler = UpDownDampingScheduler(optimizer, args.gamma)
    timer = Timer()
    for idx in range(50):
        optimizer.zero_grad()
        y = net()
        loss = y
        loss.backward()
        optimizer.step(loss)
        scheduler.step()
        print('Quad loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()), end=' ')
        print('Using Damping factor', [group['lr'] for group in optimizer.param_groups])
        if loss < 1e-7:
            print('Early Stoping with UpDownDampingScheduler!')
            print('Optimization Done with loss:', loss.item())
            break


    # Hard Test
    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(1,  6, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
            self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5, 1, 0), nn.ReLU(), nn.MaxPool2d(2))
            self.linear = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 10))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return self.linear(x)

        def parameters(self, recurse: bool = True):
            '''
            To save memory when using LM optimizer
            Better to manually group parameters according to paramter size.
            '''
            return [{'params': list(self.conv1.parameters()) + list(self.conv2.parameters())},
                    {'params': self.linear.parameters()}]


    def performance(loader, net, device):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(loader):
                if torch.cuda.is_available():
                    inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
            acc = correct/total
        return acc

    train_data = MNIST(root='/data/datasets', train=True, transform=T.ToTensor(), download=True)
    test_data = MNIST(root='/data/datasets', train=False, transform=T.ToTensor())
    train_loader = Data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    net = LeNet().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = LevenbergMarquardt(net.parameters(), damping=args.damping, max_block=2000)
    scheduler = UpDownDampingScheduler(optimizer, args.gamma, verbose=True)
    print('Testing LeNet on MNIST..')
    for idx in range(args.epoch):
        losses = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step(loss)
            losses += loss
        scheduler.step()
        acc = performance(test_loader, net, args.device)
        print('Train loss: %.7f Test acc: %.4f  @ %d epoch, '
              'Time: %.3fs'%(losses, acc, idx, timer.end()))
