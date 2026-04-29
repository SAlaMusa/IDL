#!/bin/bash
#SBATCH --job-name=conf_solarize
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2%1
#SBATCH --output=/ocean/projects/cis260134p/mkipsang/IDL/logs/conf_solarize_%A_%a.out
#SBATCH --error=/ocean/projects/cis260134p/mkipsang/IDL/logs/conf_solarize_%A_%a.err

module load anaconda3
conda activate simclr

cd /ocean/projects/cis260134p/mkipsang/IDL
mkdir -p logs results/confirmatory

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="runs/harmful_solarize_seed${SEED}"
OUT="results/confirmatory/harmful_solarize_seed${SEED}.csv"

python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print('GPU:', torch.cuda.get_device_name(0))" || exit 1

FINAL_CKPT="${RUN_NAME}/checkpoint_0800.pth.tar"
if [ -f "$FINAL_CKPT" ]; then
    echo "==> Pretraining already complete, skipping."
else
    LAST_CKPT=$(ls "${RUN_NAME}"/checkpoint_0[0-9]*.pth.tar 2>/dev/null | sort -V | grep -v 'checkpoint_0800' | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "==> Resuming from $LAST_CKPT"
        python run.py --config configs/harmful_solarize.yaml --seed $SEED --epochs 800 --run-name $RUN_NAME --resume "$LAST_CKPT"
    else
        echo "==> Starting harmful_solarize seed=$SEED from scratch"
        python run.py --config configs/harmful_solarize.yaml --seed $SEED --epochs 800 --run-name $RUN_NAME
    fi
fi

CKPT="${RUN_NAME}/checkpoint_0800.pth.tar"
echo "==> Linear eval on $CKPT"
python linear_eval.py --checkpoint $CKPT --dataset cifar10 \
    --arch resnet18 --epochs 100 -b 256 -j 4 --seed $SEED --out $OUT

EXP=$(basename $RUN_NAME)
echo "==> Alignment/uniformity (z) on $CKPT"
python analysis/compute_metrics.py --checkpoints $CKPT \
    --labels $EXP --dataset cifar10 \
    --proj-head mlp2 --out results/confirmatory/${EXP}_metrics_z.csv

echo "==> Alignment/uniformity (h) on $CKPT"
python analysis/compute_metrics.py --checkpoints $CKPT \
    --labels $EXP --dataset cifar10 \
    --proj-head none --out results/confirmatory/${EXP}_metrics_h.csv

echo "==> Done. Saved $OUT"
