#!/bin/bash
#SBATCH --job-name=conf_solarize
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
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

echo "==> Pretraining harmful_solarize seed=$SEED -> $RUN_NAME"
python run.py --config configs/harmful_solarize.yaml --seed $SEED --epochs 800 --run-name $RUN_NAME

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
