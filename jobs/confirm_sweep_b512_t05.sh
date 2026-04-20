#!/bin/bash
#SBATCH --job-name=conf_b512_t05
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --output=/ocean/projects/cis260134p/mkipsang/IDL/logs/conf_b512_t05_%A_%a.out
#SBATCH --error=/ocean/projects/cis260134p/mkipsang/IDL/logs/conf_b512_t05_%A_%a.err

module load anaconda3
conda activate simclr

cd /ocean/projects/cis260134p/mkipsang/IDL
mkdir -p logs

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running sweep_b512_t05 seed=$SEED"

python run.py --config configs/sweep_b512_t05.yaml --seed $SEED --epochs 800
