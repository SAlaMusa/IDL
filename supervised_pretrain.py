"""
Supervised pretraining baseline.

Trains ResNet-18 on CIFAR-10 with cross-entropy, then saves the backbone in the
same checkpoint format as SimCLR/MoCo (keys prefixed 'backbone.*') so the
existing linear_eval.py can load it without modification.
"""
import argparse
import os
import csv
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

from utils import save_checkpoint

parser = argparse.ArgumentParser(description='Supervised Pretraining Baseline')
parser.add_argument('-data', default='./datasets', metavar='DIR')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--fp16-precision', action='store_true')
parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument('--out-dir', default='runs/supervised', metavar='DIR')


NORMALIZE = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                   std=(0.2023, 0.1994, 0.2010))


def get_loaders(data, batch_size, workers):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), NORMALIZE])

    train_ds = datasets.CIFAR10(data, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def accuracy(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        correct = pred.t().eq(target.view(1, -1)).reshape(-1).float().sum()
        return (correct / target.size(0) * 100).item()


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    os.makedirs(args.out_dir, exist_ok=True)

    # CIFAR stem: 3×3 conv1, no maxpool
    model = models.resnet18(weights=None, num_classes=10).to(device)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = nn.Identity()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion  = nn.CrossEntropyLoss().to(device)
    scaler     = torch.amp.GradScaler('cuda', enabled=args.fp16_precision)

    train_loader, test_loader = get_loaders(args.data, args.batch_size, args.workers)

    best_top1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=args.fp16_precision):
                loss = criterion(model(images), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        top1_sum, n = 0.0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                top1_sum += accuracy(model(images), labels) * labels.size(0)
                n += labels.size(0)
        top1 = top1_sum / n  # accuracy() already returns %, weighted mean over batches

        if top1 > best_top1:
            best_top1 = top1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{args.epochs}]  Test Top-1: {top1:.2f}%  Best: {best_top1:.2f}%")

    print(f"\nBest Test Top-1: {best_top1:.2f}%")

    # Remap keys to 'backbone.*' format so linear_eval.py loads this checkpoint
    # the same way it loads SimCLR/MoCo checkpoints.
    remapped = {'backbone.' + k: v for k, v in model.state_dict().items()}

    ckpt_path = os.path.join(args.out_dir, 'supervised_ep{:04d}.pth.tar'.format(args.epochs))
    save_checkpoint(
        {'epoch': args.epochs, 'arch': args.arch, 'state_dict': remapped, 'optimizer': optimizer.state_dict()},
        is_best=False,
        filename=ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")

    results_path = os.path.join(args.out_dir, 'supervised_results.csv')
    with open(results_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['exp_name', 'epochs', 'best_top1', 'checkpoint', 'timestamp'])
        w.writerow(['supervised', args.epochs, f'{best_top1:.2f}', ckpt_path,
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M')])
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
