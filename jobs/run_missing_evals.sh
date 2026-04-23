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

    # skip if no completed checkpoint or config
    if [ ! -f "$CKPT" ] || [ ! -f "$CONFIG" ]; then
        continue
    fi

    # skip if checkpoint is too small to be valid (< 10MB)
    CKPT_SIZE=$(stat -c%s "$CKPT" 2>/dev/null || echo 0)
    if [ "$CKPT_SIZE" -lt 10000000 ]; then
        echo "SKIP — checkpoint too small ($CKPT_SIZE bytes), likely incomplete: $RUN_DIR"
        continue
    fi

    # extract experiment info from config.yml
    DATASET=$(python -c "
import yaml, sys
c = yaml.safe_load(open('$CONFIG'))
if isinstance(c, dict):
    print(c.get('dataset_name', 'cifar10'))
else:
    print(getattr(c, 'dataset_name', 'cifar10'))
" 2>/dev/null || echo "cifar10")

    SEED=$(python -c "
import yaml, sys
c = yaml.safe_load(open('$CONFIG'))
if isinstance(c, dict):
    print(c.get('seed', 42))
else:
    print(getattr(c, 'seed', 42))
" 2>/dev/null || echo "42")

    EXP=$(python -c "
import yaml, os
c = yaml.safe_load(open('$CONFIG'))
if isinstance(c, dict):
    cfg = c.get('config', '')
else:
    cfg = getattr(c, 'config', '')
if cfg:
    print(os.path.splitext(os.path.basename(str(cfg)))[0])
else:
    print(os.path.basename('$RUN_DIR'.rstrip('/')))
" 2>/dev/null || echo "unknown")

    OUT="results/confirmatory/${EXP}_seed${SEED}.csv"

    # skip if we already have this result
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
