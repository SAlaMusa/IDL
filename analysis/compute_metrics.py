"""
Alignment & Uniformity metrics (Wang & Isola, 2020) for trained SimCLR / MoCo checkpoints.

Alignment:  E_{pos pairs} [ ||f(x) - f(x')||_2^2 ]          (lower = more aligned)
Uniformity: log E_{x,y}   [ exp(-2 * ||f(x) - f(y)||_2^2) ] (lower = more uniform)

Both metrics use L2-normalised projection-head outputs.

Usage:
    python analysis/compute_metrics.py \
        --checkpoints results/kaggle_cifar10_baseline/cifar10_resnet18_ep200_seed42.pth.tar \
                      results/kaggle_session_a_ablations/ablation_no_blur_ep200.pth.tar \
        --labels baseline ablation_no_blur \
        --out results/alignment_uniformity.csv
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_simclr import ResNetSimCLR

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', nargs='+', required=True,
                    help='one or more .pth.tar checkpoint paths')
parser.add_argument('--labels', nargs='+', default=None,
                    help='short names for each checkpoint (defaults to filename stem)')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10'])
parser.add_argument('-data', default='./datasets')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--out_dim', default=128, type=int)
parser.add_argument('--proj-head', default='mlp2',
                    choices=['none', 'linear', 'mlp2', 'mlp3'])
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('--n-batches', default=20, type=int,
                    help='number of batches to use for metric estimation (0 = full dataset)')
parser.add_argument('--out', default='results/alignment_uniformity.csv')


# Two independent random crops so we get positive pairs without needing the
# full contrastive pipeline — same distribution used during SimCLR training.
def _aug(dataset):
    if dataset == 'cifar10':
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        crop_size = 32
    else:
        norm = transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2242, 0.2215, 0.2239))
        crop_size = 96

    t = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        norm,
    ])
    return t


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return two augmented views of each image."""
    def __init__(self, base_ds, transform):
        self.base = base_ds
        self.t = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.t(img), self.t(img)


def get_dataset(name, root, transform):
    if name == 'cifar10':
        return datasets.CIFAR10(root, train=True, download=True, transform=None)
    return datasets.STL10(root, split='unlabeled', download=True, transform=None)


@torch.no_grad()
def compute(model, loader, device, n_batches):
    model.eval()
    z1_all, z2_all = [], []
    for i, (x1, x2) in enumerate(loader):
        if n_batches > 0 and i >= n_batches:
            break
        z1 = F.normalize(model(x1.to(device)), dim=1)
        z2 = F.normalize(model(x2.to(device)), dim=1)
        z1_all.append(z1.cpu())
        z2_all.append(z2.cpu())

    z1 = torch.cat(z1_all)   # (N, dim)
    z2 = torch.cat(z2_all)

    alignment = (z1 - z2).norm(dim=1).pow(2).mean().item()

    z = torch.cat([z1, z2])  # pool both views for uniformity
    sq_pdist = torch.pdist(z, p=2).pow(2)
    uniformity = sq_pdist.mul(-2).exp().mean().log().item()

    return alignment, uniformity


def load_model(ckpt_path, arch, out_dim, proj_head, cifar_stem, device):
    model = ResNetSimCLR(base_model=arch, out_dim=out_dim,
                         cifar_stem=cifar_stem, proj_head=proj_head).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['state_dict']
    # strict=False so that supervised checkpoints (backbone.fc has wrong shape) load partially
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")
    return model


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cifar_stem = (args.dataset == 'cifar10')

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0]
                              for p in args.checkpoints]
    assert len(labels) == len(args.checkpoints), \
        "--labels count must match --checkpoints count"

    t = _aug(args.dataset)
    base_ds = get_dataset(args.dataset, args.data, transform=None)
    ds = TwoViewDataset(base_ds, t)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    rows = []

    for label, ckpt_path in zip(labels, args.checkpoints):
        print(f"\n=> {label}  ({ckpt_path})")
        model = load_model(ckpt_path, args.arch, args.out_dim,
                           args.proj_head, cifar_stem, device)
        align, unif = compute(model, loader, device, args.n_batches)
        print(f"   alignment={align:.4f}   uniformity={unif:.4f}")
        rows.append({'label': label, 'alignment': f'{align:.4f}',
                     'uniformity': f'{unif:.4f}', 'checkpoint': ckpt_path})

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['label', 'alignment', 'uniformity', 'checkpoint'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
