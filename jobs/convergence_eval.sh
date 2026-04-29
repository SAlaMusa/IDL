#!/bin/bash
#SBATCH --job-name=convergence
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-11
#SBATCH --output=/ocean/projects/cis260134p/mkipsang/IDL/logs/convergence_%A_%a.out
#SBATCH --error=/ocean/projects/cis260134p/mkipsang/IDL/logs/convergence_%A_%a.err

# Runs linear eval at epochs 200/400/600 for the 4 convergence-curve experiments.
# Submit AFTER the main confirmatory jobs finish.
# Each array task handles one experiment+seed combination (3 epochs serially ~1h total).

module load anaconda3
conda activate simclr

cd /ocean/projects/cis260134p/mkipsang/IDL
mkdir -p logs results/confirmatory

python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print('GPU:', torch.cuda.get_device_name(0))" || exit 1

# 4 experiments × 3 seeds = 12 array tasks
EXPS=(
    baseline_cifar10   baseline_cifar10   baseline_cifar10
    ablation_no_crop   ablation_no_crop   ablation_no_crop
    harmful_solarize   harmful_solarize   harmful_solarize
    pair_jitter_grayscale pair_jitter_grayscale pair_jitter_grayscale
)
SEEDS=(42 43 44 42 43 44 42 43 44 42 43 44)

EXP=${EXPS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="runs/${EXP}_seed${SEED}"

DATASET="cifar10"
[[ "$EXP" == *"stl10"* ]] && DATASET="stl10"

echo "==> Convergence eval: $EXP seed=$SEED"

for EP in 200 400 600; do
    CKPT="${RUN_NAME}/checkpoint_$(printf '%04d' $EP).pth.tar"
    OUT="results/confirmatory/${EXP}_seed${SEED}_ep${EP}.csv"

    if [ ! -f "$CKPT" ]; then
        echo "  SKIP ep=$EP — checkpoint not found: $CKPT"
        continue
    fi

    if [ -f "$OUT" ]; then
        echo "  SKIP ep=$EP — already evaluated: $OUT"
        continue
    fi

    echo "  ==> Linear eval ep=$EP on $CKPT"
    python linear_eval.py --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --arch resnet18 --epochs 100 -b 256 -j 4 --seed $SEED --out "$OUT"
done

echo "==> Done: $EXP seed=$SEED"
