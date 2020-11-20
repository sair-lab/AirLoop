import torch
from torch.optim import Optimizer


class LM(Optimizer):
    '''
    Levenberg–Marquardt algorithm
    args:
    df (damping factor): scalar

    (J^T J + df I) * Delta = J^T loss
    '''
    def __init__(self, params, df):
        defaults = dict(df=df)
        super().__init__(params, defaults)
        if df < 0.0:
            raise ValueError("Invalid damping factor: {}".format(df))

    @torch.no_grad()
    def step(self, loss):
        for group in self.param_groups:
            numels = [p.numel() for p in group['params'] if p.grad is not None]
            J = torch.cat([p.grad.data.view(1,-1) for p in group['params'] if p.grad is not None],-1)
            A = (J.T @ J) + group['df'] * torch.eye(J.size(-1)).to(J)
            D = J.T.cholesky_solve(A.cholesky()).split(numels)
            [p.add_(d.view(p.shape), alpha=-loss) for p,d in zip(group['params'], D) if p.grad is not None]


if __name__ == "__main__":
    '''
    Test Optimizer
    '''
    import argparse
    from torch import nn
    import torch.utils.data as Data
    from torchvision.datasets import MNIST
    from torchvision import transforms as T

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--epoch', default=20, type=int, help='epoch')
    parser.add_argument('--batch-size', default=1000, type=int, help='epoch')
    parser.add_argument('--df', default=0.1, type=float, help='Damping factor')
    args = parser.parse_args()

    # Easy Test
    class QuadNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(10,10))

        def forward(self):
            return (self.w**2).sum()

    net = QuadNet().to(args.device)
    optimizer = LM(net.parameters(), df=args.df)

    print('Testing Quadratic function...')
    for idx in range(1000):
        loss = net()
        loss.backward()
        optimizer.step(loss.item())
        print('Quadratic loss %.4f @ %d iteration'%(loss, idx))


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
    optimizer = LM(net.parameters(), df=args.df)

    print('Testing LeNet on MNIST..')
    for idx in range(args.epoch):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step(loss.item())
        acc = performance(test_loader, net, args.device)
        print('MNIST acc: %.4f @ %d epoch'%(acc, idx))
