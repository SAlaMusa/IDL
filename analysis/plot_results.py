"""
Generate all result figures for the SimCLR ablation study.

Reads CSVs from results/ and writes PNGs to plots/.

Figures produced:
  1. fig1_single_ablations.png   — single augmentation ablations vs baseline
  2. fig2_pairwise_ablations.png — pairwise ablations vs baseline
  3. fig3_harmful_augs.png       — harmful augmentations vs baseline
  4. fig4_proj_head.png          — projection head variants
  5. fig5_batch_temp_heatmap.png — batch × temperature sweep heatmap
  6. fig6_alignment_uniformity.png (optional, requires results/alignment_uniformity.csv)

Usage:
    python analysis/plot_results.py
    python analysis/plot_results.py --out-dir my_plots/
"""
import argparse
import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', default='results')
parser.add_argument('--out-dir', default='plots')


# ── helpers ──────────────────────────────────────────────────────────────────

STYLE = {
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
}

BLUE    = '#4C72B0'
RED     = '#DD4444'
GREEN   = '#2CA02C'
GRAY    = '#888888'
ORANGE  = '#FF7F0E'

BASELINE_CIFAR10 = 76.36
BASELINE_STL10   = 68.03


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def hbar(ax, names, values, baseline, colors=None):
    """Horizontal bar chart with a vertical baseline line."""
    y = np.arange(len(names))
    if colors is None:
        colors = [RED if v < baseline else GREEN if v > baseline else BLUE
                  for v in values]
    bars = ax.barh(y, values, color=colors, height=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(baseline, color=GRAY, linewidth=1.5, linestyle='--', label=f'Baseline {baseline:.2f}%')
    ax.set_xlabel('Linear Eval Top-1 (%)', fontsize=10)
    for bar, val in zip(bars, values):
        offset = 0.2 if val >= baseline else -0.2
        ha = 'left' if val >= baseline else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', ha=ha, fontsize=8)
    ax.legend(fontsize=8)
    xlim = ax.get_xlim()
    ax.set_xlim(max(0, min(values) - 5), max(values) + 5)


# ── figure 1: single ablations ───────────────────────────────────────────────

def fig_single_ablations(out_dir):
    data = [
        ('Baseline (all augs)',   BASELINE_CIFAR10),
        ('No crop',               48.97),
        ('No flip',               75.73),
        ('No jitter',             70.83),
        ('No grayscale',          72.61),
        ('No blur',               76.71),
    ]
    names  = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [BLUE] + [RED if v < BASELINE_CIFAR10 else GREEN for v in values[1:]]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        hbar(ax, names, values, BASELINE_CIFAR10, colors)
        ax.set_title('Single Augmentation Ablations — CIFAR-10 (200 ep)', fontsize=11)
        savefig(fig, os.path.join(out_dir, 'fig1_single_ablations.png'))


# ── figure 2: pairwise ablations ─────────────────────────────────────────────

def fig_pairwise_ablations(out_dir):
    data = [
        ('Crop + Jitter',         74.22),
        ('Crop + Grayscale',      72.33),
        ('Crop + Blur',           59.03),
        ('Jitter + Grayscale',    41.87),
    ]
    names  = [d[0] for d in data]
    values = [d[1] for d in data]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        hbar(ax, names, values, BASELINE_CIFAR10)
        ax.set_title('Pairwise Augmentation Ablations — CIFAR-10 (200 ep)', fontsize=11)
        savefig(fig, os.path.join(out_dir, 'fig2_pairwise_ablations.png'))


# ── figure 3: harmful augmentations ──────────────────────────────────────────

def fig_harmful(out_dir):
    data = [
        ('Baseline (all augs)',  BASELINE_CIFAR10),
        ('+ Rotation 180°',      65.53),
        ('+ Gaussian Noise',     62.42),
        ('+ Solarize',           78.31),
    ]
    names  = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [BLUE, RED, RED, GREEN]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        hbar(ax, names, values, BASELINE_CIFAR10, colors)
        ax.set_title('Harmful Augmentation Study — CIFAR-10 (200 ep)', fontsize=11)
        savefig(fig, os.path.join(out_dir, 'fig3_harmful_augs.png'))


# ── figure 4: projection head ─────────────────────────────────────────────────

def fig_proj_head(out_dir):
    data = [
        ('None (linear)',  76.40),
        ('Linear',         76.65),
        ('MLP-2 (default)', BASELINE_CIFAR10),
        ('MLP-3',          76.15),
    ]
    names  = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [BLUE if n == 'MLP-2 (default)' else ORANGE for n in names]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        hbar(ax, names, values, BASELINE_CIFAR10, colors)
        ax.set_xlim(74, 77.5)
        ax.set_title('Projection Head Variants — CIFAR-10 (200 ep)', fontsize=11)
        savefig(fig, os.path.join(out_dir, 'fig4_proj_head.png'))


# ── figure 5: batch × temperature heatmap ─────────────────────────────────────

def fig_heatmap(results_dir, out_dir):
    batches = [64, 128, 256, 512]
    temps   = [0.1, 0.2, 0.5, 1.0]

    # fill from CSVs
    grid = np.full((len(batches), len(temps)), np.nan)
    sweep_dir = os.path.join(results_dir, 'batch_temp_sweep')
    for fname in os.listdir(sweep_dir):
        if not fname.endswith('.csv'):
            continue
        with open(os.path.join(sweep_dir, fname)) as f:
            for row in csv.DictReader(f):
                b = int(row['batch'])
                t = float(row['temperature'])
                v = float(row['best_top1'])
                if b in batches and t in temps:
                    bi = batches.index(b)
                    ti = temps.index(t)
                    grid[bi, ti] = v

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vmin = np.nanmin(grid) - 1
        vmax = np.nanmax(grid) + 1
        im = ax.imshow(grid, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([str(t) for t in temps])
        ax.set_yticks(range(len(batches)))
        ax.set_yticklabels([str(b) for b in batches])
        ax.set_xlabel('Temperature τ', fontsize=10)
        ax.set_ylabel('Batch Size', fontsize=10)
        ax.set_title('Batch × Temperature Sweep — Top-1 Accuracy (%)', fontsize=11)
        plt.colorbar(im, ax=ax, label='Top-1 (%)')
        for bi in range(len(batches)):
            for ti in range(len(temps)):
                v = grid[bi, ti]
                if not np.isnan(v):
                    ax.text(ti, bi, f'{v:.1f}', ha='center', va='center',
                            fontsize=8, color='black')
                else:
                    ax.text(ti, bi, '—', ha='center', va='center',
                            fontsize=10, color='gray')
        savefig(fig, os.path.join(out_dir, 'fig5_batch_temp_heatmap.png'))


# ── figure 6: alignment / uniformity scatter ──────────────────────────────────

def fig_alignment_uniformity(results_dir, out_dir):
    path = os.path.join(results_dir, 'alignment_uniformity.csv')
    if not os.path.exists(path):
        print(f"  skipping fig6 (not found: {path})")
        return

    with open(path) as f:
        rows = list(csv.DictReader(f))

    labels     = [r['label']      for r in rows]
    alignments = [float(r['alignment'])   for r in rows]
    uniformities = [float(r['uniformity']) for r in rows]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(alignments, uniformities, color=BLUE, s=60, zorder=3)
        for lbl, a, u in zip(labels, alignments, uniformities):
            ax.annotate(lbl, (a, u), textcoords='offset points',
                        xytext=(5, 3), fontsize=7)
        ax.set_xlabel('Alignment ↓', fontsize=10)
        ax.set_ylabel('Uniformity ↓', fontsize=10)
        ax.set_title('Alignment vs Uniformity (projection-head outputs)', fontsize=11)
        savefig(fig, os.path.join(out_dir, 'fig6_alignment_uniformity.png'))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Writing plots to {args.out_dir}/")

    fig_single_ablations(args.out_dir)
    fig_pairwise_ablations(args.out_dir)
    fig_harmful(args.out_dir)
    fig_proj_head(args.out_dir)
    fig_heatmap(args.results_dir, args.out_dir)
    fig_alignment_uniformity(args.results_dir, args.out_dir)

    print("Done.")


if __name__ == '__main__':
    main()
