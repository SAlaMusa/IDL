#!/bin/bash
#SBATCH --job-name=conf_no_crop
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --output=/ocean/projects/cis260134p/mkipsang/IDL/logs/conf_no_crop_%A_%a.out
#SBATCH --error=/ocean/projects/cis260134p/mkipsang/IDL/logs/conf_no_crop_%A_%a.err

module load anaconda3
conda activate simclr

cd /ocean/projects/cis260134p/mkipsang/IDL
mkdir -p logs results/confirmatory

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "==> Pretraining ablation_no_crop seed=$SEED"
BEFORE=$(ls -d runs/*/ 2>/dev/null | sort)
python run.py --config configs/ablation_no_crop.yaml --seed $SEED --epochs 800
RUN_DIR=$(comm -13 <(echo "$BEFORE") <(ls -d runs/*/ 2>/dev/null | sort) | head -1)
CKPT="${RUN_DIR}checkpoint_0800.pth.tar"

echo "==> Linear eval on $CKPT"
python linear_eval.py --checkpoint $CKPT --dataset cifar10 \
    --arch resnet18 --epochs 100 -b 256 -j 4 --seed $SEED

cp linear_eval_results.csv results/confirmatory/ablation_no_crop_seed${SEED}.csv
echo "==> Done. Saved results/confirmatory/ablation_no_crop_seed${SEED}.csv"
