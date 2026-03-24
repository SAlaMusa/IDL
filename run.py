import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from optimizers.lars import LARS
from simclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--config', default=None, metavar='PATH',
                    help='path to a YAML config file (e.g. configs/baseline_cifar10.yaml). '
                         'Any CLI argument will override the value in the config file.')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='number of linear warmup epochs before cosine decay')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR',
                    help='base learning rate. If not set, uses the scaled formula '
                         'lr = 0.3 * batch_size / 256 (as per proposal Section 4.2)',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--proj-head', default='mlp2', choices=['none', 'linear', 'mlp2', 'mlp3'],
                    help='Projection head type (default: mlp2)')


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # First pass: grab --config before full parsing so we can set defaults
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', default=None)
    pre_args, _ = pre_parser.parse_known_args()

    aug_cfg = {}
    if pre_args.config is not None:
        config = load_config(pre_args.config)
        aug_cfg = config.get('aug', {}) or {}
        parser.set_defaults(**{
            'dataset_name':      config.get('dataset_name', 'stl10'),
            'arch':              config.get('arch', 'resnet18'),
            'epochs':            config.get('epochs', 800),
            'warmup_epochs':     config.get('warmup_epochs', 10),
            'batch_size':        config.get('batch_size', 512),
            'lr':                config.get('lr', None),
            'weight_decay':      config.get('weight_decay', 1e-6),
            'temperature':       config.get('temperature', 0.07),
            'seed':              config.get('seed', None),
            'out_dim':           config.get('out_dim', 128),
            'n_views':           config.get('n_views', 2),
            'workers':           config.get('workers', 2),
            'fp16_precision':    config.get('fp16_precision', False),
            'log_every_n_steps': config.get('log_every_n_steps', 100),
            'proj_head':         config.get('proj_head', 'mlp2'),
        })

    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # Scaled learning rate: lr = 0.3 * batch_size / 256  (proposal Section 4.2)
    if args.lr is None:
        args.lr = 0.3 * args.batch_size / 256
        print(f"=> Using scaled learning rate: {args.lr:.4f}  "
              f"(= 0.3 × {args.batch_size} / 256)")

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)
    if aug_cfg:
        active = [k for k, v in aug_cfg.items() if isinstance(v, bool)]
        print(f"=> Aug recipe: {aug_cfg}")
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, aug_cfg=aug_cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Use CIFAR stem (3x3 conv, no maxpool) for 32x32 CIFAR-10 images
    cifar_stem = (args.dataset_name == 'cifar10')
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, cifar_stem=cifar_stem,
                         proj_head=args.proj_head)

    # LARS optimizer as specified in proposal Section 4.2
    optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                     momentum=0.9, eta=0.001)

    # Cosine annealing over full training (warmup handled inside SimCLR.train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

    #  It's a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
