"""
Plot linear evaluation accuracy vs. training epoch (convergence curves).

Reads CSVs produced by:
  - jobs/confirm_*.sh        → results/confirmatory/<exp>_seed<N>.csv             (epoch 800)
  - jobs/convergence_eval.sh → results/confirmatory/<exp>_seed<N>_ep<E>.csv       (epochs 200/400/600)

Usage:
    python analysis/plot_convergence.py
    python analysis/plot_convergence.py --results-dir results/confirmatory --out plots/convergence.png
"""
import argparse
import os
import re
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', default='results/confirmatory')
parser.add_argument('--out', default='plots/convergence.png')
parser.add_argument('--experiments', nargs='+',
                    default=['baseline_cifar10', 'ablation_no_crop',
                             'harmful_solarize', 'pair_jitter_grayscale'],
                    help='base experiment names to include (without _seed* suffix)')

LABELS = {
    'baseline_cifar10':    'Full pipeline',
    'ablation_no_crop':    'No crop',
    'harmful_solarize':    'Solarize',
    'pair_jitter_grayscale': 'Jitter + Grayscale',
}


def _read_top1(csv_path):
    """Return the best_top1 value from a linear_eval results CSV (last data row)."""
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]['best_top1'])


def collect(results_dir, experiments):
    """
    Returns dict: {exp_base: {epoch: [acc_seed1, acc_seed2, ...]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    # ── intermediate epochs: <exp>_ep<E>.csv  (EXP encodes seed, e.g. baseline_cifar10_seed42)
    inter_re = re.compile(r'^(.+)_ep(\d+)\.csv$')
    # ── final epoch: <exp>_seed<N>.csv  (no _ep, no _metrics)
    final_re = re.compile(r'^(.+)_seed\d+\.csv$')

    for fname in os.listdir(results_dir):
        path = os.path.join(results_dir, fname)

        m = inter_re.match(fname)
        if m:
            full_exp, epoch = m.group(1), int(m.group(2))
            # strip _seed<N> from full_exp to get base name
            base = re.sub(r'_seed\d+$', '', full_exp)
            if base in experiments:
                acc = _read_top1(path)
                if acc is not None:
                    data[base][epoch].append(acc)
            continue

        m = final_re.match(fname)
        if m and '_metrics_' not in fname and '_ep' not in fname:
            full_exp = m.group(1)
            base = re.sub(r'_seed\d+$', '', full_exp)
            if base in experiments:
                acc = _read_top1(path)
                if acc is not None:
                    data[base][800].append(acc)

    return data


def main():
    args = parser.parse_args()
    data = collect(args.results_dir, args.experiments)

    if not data:
        print("No convergence data found. Run jobs/convergence_eval.sh first.")
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors

    for i, exp in enumerate(args.experiments):
        if exp not in data:
            print(f"  [skip] no data for {exp}")
            continue

        epoch_data = data[exp]
        epochs = sorted(epoch_data.keys())
        means = [np.mean(epoch_data[e]) for e in epochs]
        stds  = [np.std(epoch_data[e])  for e in epochs]

        label = LABELS.get(exp, exp)
        color = colors[i % len(colors)]
        ax.plot(epochs, means, marker='o', label=label, color=color)
        ax.fill_between(epochs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

        n_seeds = max(len(v) for v in epoch_data.values())
        print(f"  {label}: epochs={epochs}, means={[f'{m:.1f}' for m in means]} ({n_seeds} seeds)")

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Linear eval top-1 accuracy (%)')
    ax.set_title('Convergence curves — SimCLR on CIFAR-10')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([200, 400, 600, 800])

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
