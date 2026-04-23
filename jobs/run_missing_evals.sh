#!/bin/bash
#SBATCH --job-name=missing_evals
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/ocean/projects/cis260134p/mkipsang/IDL/logs/missing_evals_%j.out
#SBATCH --error=/ocean/projects/cis260134p/mkipsang/IDL/logs/missing_evals_%j.err

module load anaconda3
conda activate simclr

cd /ocean/projects/cis260134p/mkipsang/IDL
mkdir -p results/confirmatory

for RUN_DIR in runs/*/; do
    CKPT="${RUN_DIR}checkpoint_0800.pth.tar"
    CONFIG="${RUN_DIR}config.yml"

    # skip if no completed checkpoint
    if [ ! -f "$CKPT" ]; then
        continue
    fi

    # skip if checkpoint is too small to be valid (< 10MB)
    CKPT_SIZE=$(stat -c%s "$CKPT" 2>/dev/null || echo 0)
    if [ "$CKPT_SIZE" -lt 10000000 ]; then
        echo "SKIP — checkpoint too small ($CKPT_SIZE bytes), likely incomplete: $RUN_DIR"
        continue
    fi

    # identify experiment from training.log
    LOG="${RUN_DIR}training.log"
    DATASET="cifar10"
    SEED="unknown"

    if [ -f "$LOG" ]; then
        # check if STL-10 (larger image size affects dataset)
        grep -q "stl10" "$LOG" 2>/dev/null && DATASET="stl10"
        # extract seed from log if present
        SEED_LINE=$(grep -i "seed" "$LOG" 2>/dev/null | head -1)
    fi

    # use run directory name as experiment identifier
    EXP=$(basename "$RUN_DIR")
    OUT="results/confirmatory/eval_${EXP}.csv"

    # skip if already evaluated
    if [ -f "$OUT" ]; then
        echo "SKIP — already done: $OUT"
        continue
    fi

    echo "==> Evaluating $EXP | dataset=$DATASET | seed=$SEED"
    echo "    checkpoint: $CKPT"

    python linear_eval.py \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --arch resnet18 \
        --epochs 100 \
        -b 256 -j 4 \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        cp linear_eval_results.csv "$OUT"
        echo "    Saved: $OUT"
    else
        echo "    ERROR: linear eval failed for $CKPT"
    fi
done

echo ""
echo "==> All done. Results in results/confirmatory/:"
ls results/confirmatory/
