import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from optimizers.lars import LARS
from moco import MoCo

parser = argparse.ArgumentParser(description='PyTorch MoCo v2')
parser.add_argument('--config', default=None, metavar='PATH',
                    help='path to a YAML config file')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--warmup-epochs', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--lr', '--learning-rate', default=None, type=float, dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float, dest='weight_decay')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--fp16-precision', action='store_true')
parser.add_argument('--out_dim', default=128, type=int)
parser.add_argument('--log-every-n-steps', default=100, type=int)
parser.add_argument('--temperature', default=0.07, type=float)
parser.add_argument('--n-views', default=2, type=int)
parser.add_argument('--gpu-index', default=0, type=int)
parser.add_argument('--moco-queue-size', default=4096, type=int,
                    help='MoCo queue size (number of negative keys)')
parser.add_argument('--moco-momentum', default=0.999, type=float,
                    help='MoCo key encoder EMA momentum')
parser.add_argument('--run-name', default=None,
                    help='Custom output directory for this run.')


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', default=None)
    pre_args, _ = pre_parser.parse_known_args()

    aug_cfg = {}
    if pre_args.config is not None:
        cfg = load_config(pre_args.config)
        aug_cfg = cfg.get('aug', {}) or {}
        parser.set_defaults(**{
            'dataset_name':      cfg.get('dataset_name', 'cifar10'),
            'arch':              cfg.get('arch', 'resnet18'),
            'epochs':            cfg.get('epochs', 200),
            'warmup_epochs':     cfg.get('warmup_epochs', 10),
            'batch_size':        cfg.get('batch_size', 128),
            'lr':                cfg.get('lr', None),
            'weight_decay':      cfg.get('weight_decay', 1e-6),
            'temperature':       cfg.get('temperature', 0.07),
            'seed':              cfg.get('seed', None),
            'out_dim':           cfg.get('out_dim', 128),
            'n_views':           cfg.get('n_views', 2),
            'workers':           cfg.get('workers', 2),
            'fp16_precision':    cfg.get('fp16_precision', False),
            'log_every_n_steps': cfg.get('log_every_n_steps', 100),
            'moco_queue_size':   cfg.get('moco_queue_size', 4096),
            'moco_momentum':     cfg.get('moco_momentum', 0.999),
        })

    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.3 * args.batch_size / 256
        print(f"=> Scaled LR: {args.lr:.4f}  (= 0.3 × {args.batch_size} / 256)")

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.seed is not None:
        torch.manual_seed(args.seed)

    dataset = ContrastiveLearningDataset(args.data)
    if aug_cfg:
        print(f"=> Aug recipe: {aug_cfg}")
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, aug_cfg=aug_cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    cifar_stem = (args.dataset_name == 'cifar10')
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim,
                         cifar_stem=cifar_stem, proj_head='mlp2')

    optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                     momentum=0.9, eta=0.001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        moco = MoCo(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        moco.train(train_loader)


if __name__ == "__main__":
    main()
