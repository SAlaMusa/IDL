import argparse
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='SimCLR Linear Evaluation')
parser.add_argument('--checkpoint', required=True, metavar='PATH',
                    help='path to pretrained SimCLR checkpoint (.pth.tar)')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10'],
                    help='dataset to evaluate on')
parser.add_argument('-data', default='./datasets', metavar='DIR',
                    help='path to dataset root (same as used during pretraining)')
parser.add_argument('-a', '--arch', default='resnet18', choices=['resnet18', 'resnet50'],
                    help='encoder architecture (must match the pretrained checkpoint)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs to train the linear classifier')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='batch size for linear evaluation')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate for the linear classifier (SGD, default 0.1)')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--seed', default=42, type=int,
                    help='random seed for reproducibility')
parser.add_argument('--disable-cuda', action='store_true',
                    help='disable CUDA even if available')


def get_data_loaders(dataset_name, data_path, batch_size, workers):
    """
    Load labeled train and test splits for linear evaluation.
    Applies standard per-channel normalization as specified in proposal Section 4.2.
    No other augmentation — we evaluate frozen representations as-is.
    """
    # Per-channel mean and std for standard normalization
    NORMALIZE = {
        'cifar10': transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                        std=(0.2023, 0.1994, 0.2010)),
        'stl10':   transforms.Normalize(mean=(0.4467, 0.4398, 0.4066),
                                        std=(0.2242, 0.2215, 0.2239)),
    }
    transform = transforms.Compose([transforms.ToTensor(), NORMALIZE[dataset_name]])

    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset  = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    else:  # stl10
        train_dataset = datasets.STL10(data_path, split='train', download=True, transform=transform)
        test_dataset  = datasets.STL10(data_path, split='test',  download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, drop_last=False, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size * 2, shuffle=False,
                              num_workers=workers, drop_last=False, pin_memory=True)
    return train_loader, test_loader


def load_encoder(arch, checkpoint_path, device, num_classes=10):
    """
    Build a standard ResNet, then load the SimCLR pretrained encoder weights into it.

    The SimCLR checkpoint stores the model as ResNetSimCLR, which has:
      - backbone.conv1, backbone.layer1 ... backbone.layer4  (encoder)
      - backbone.fc.0, backbone.fc.2                         (projection head — discard)

    We strip the 'backbone.' prefix and skip 'backbone.fc.*' so only the
    convolutional encoder is loaded. The fresh fc layer (classifier) is left
    randomly initialized and will be the only thing we train.
    """
    if arch == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    else:
        model = models.resnet50(weights=None, num_classes=num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # Remap keys: 'backbone.layer1.0.conv1.weight' -> 'layer1.0.conv1.weight'
    # Skip:       'backbone.fc.*'  (projection head)
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.') and not k.startswith('backbone.fc.'):
            encoder_state[k[len('backbone.'):]] = v

    log = model.load_state_dict(encoder_state, strict=False)

    # Only fc.weight and fc.bias should be missing (the fresh classifier head)
    assert set(log.missing_keys) == {'fc.weight', 'fc.bias'}, \
        f"Unexpected missing keys: {log.missing_keys}"
    assert len(log.unexpected_keys) == 0, \
        f"Unexpected keys in checkpoint: {log.unexpected_keys}"

    return model


def accuracy(output, target, topk=(1,)):
    """Top-k accuracy averaged over the batch."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print(f"=> Loading checkpoint: {args.checkpoint}")
    print(f"=> Dataset: {args.dataset}  |  Arch: {args.arch}  |  Device: {device}")

    model = load_encoder(args.arch, args.checkpoint, device, num_classes=10)

    # Freeze everything except the linear classifier head
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(trainable) == 2, "Expected exactly fc.weight and fc.bias to be trainable"
    print(f"=> Encoder frozen. Training only fc layer "
          f"({sum(p.numel() for p in trainable):,} parameters)")

    # SGD with momentum 0.9 as specified in proposal Section 4.2
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_data_loaders(
        args.dataset, args.data, args.batch_size, args.workers)

    best_top1 = 0.0

    for epoch in range(args.epochs):
        # --- Train linear head ---
        model.train()
        top1_train_sum = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            top1_train_sum += accuracy(logits, labels, topk=(1,))[0].item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        top1_train = top1_train_sum / (i + 1)

        # --- Evaluate on test set ---
        model.eval()
        top1_test_sum = 0.0
        top5_test_sum = 0.0
        with torch.no_grad():
            for j, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                top1_test_sum += top1.item()
                top5_test_sum += top5.item()
        top1_test = top1_test_sum / (j + 1)
        top5_test = top5_test_sum / (j + 1)

        if top1_test > best_top1:
            best_top1 = top1_test

        print(f"Epoch [{epoch + 1:3d}/{args.epochs}]  "
              f"Train Top-1: {top1_train:.2f}%  |  "
              f"Test Top-1: {top1_test:.2f}%  "
              f"Test Top-5: {top5_test:.2f}%")

    print(f"\nBest Test Top-1 Accuracy: {best_top1:.2f}%")


if __name__ == '__main__':
    main()
