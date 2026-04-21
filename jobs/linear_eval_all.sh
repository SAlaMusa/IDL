#!/bin/bash
#SBATCH --job-name=linear_eval
#SBATCH --account=cis260134p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/ocean/projects/cis260134p/mkipsang/IDL/logs/linear_eval_all_%j.out
#SBATCH --error=/ocean/projects/cis260134p/mkipsang/IDL/logs/linear_eval_all_%j.err

module load anaconda3
conda activate simclr

cd /ocean/projects/cis260134p/mkipsang/IDL
mkdir -p results/confirmatory

for RUN_DIR in runs/*/; do
    CKPT="${RUN_DIR}checkpoint_0800.pth.tar"
    CONFIG="${RUN_DIR}config.yml"

    # skip if checkpoint doesn't exist (incomplete run)
    if [ ! -f "$CKPT" ]; then
        echo "Skipping $RUN_DIR — no checkpoint found"
        continue
    fi

    # read dataset and config name from config.yml
    DATASET=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('dataset_name','cifar10'))")
    SEED=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('seed', 42))")

    # derive experiment name from the config file path stored in config.yml
    EXP=$(python -c "
import yaml, os
c = yaml.safe_load(open('$CONFIG'))
# config.yml stores args as a namespace — get the config path
cfg = getattr(c, 'config', None) or c.get('config', '$RUN_DIR')
name = os.path.splitext(os.path.basename(str(cfg)))[0] if cfg else os.path.basename('$RUN_DIR'.rstrip('/'))
print(name)
")

    OUT="results/confirmatory/${EXP}_seed${SEED}.csv"

    if [ -f "$OUT" ]; then
        echo "Skipping $EXP seed=$SEED — already evaluated"
        continue
    fi

    echo "==> Evaluating $EXP (dataset=$DATASET, seed=$SEED)"
    python linear_eval.py \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --arch resnet18 \
        --epochs 100 \
        -b 256 -j 4 \
        --seed "$SEED"

    cp linear_eval_results.csv "$OUT"
    echo "    Saved to $OUT"
done

echo "==> All done."
