#!/bin/bash
#SBATCH --job-name=dr-crossval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/crossval_%j.out
#SBATCH --error=logs/slurm/crossval_%j.err
#SBATCH --mail-type=END,FAIL

# Northeastern Explorer cluster modules
module load cuda/12.8.0
module load python/3.13.5

source venv/bin/activate
mkdir -p logs/slurm

# Override via: sbatch --export=CONFIG=configs/train_dinov2.yaml,FOLDS=3 slurm_crossval.sh
CONFIG=${CONFIG:-configs/train_efficientnet.yaml}
FOLDS=${FOLDS:-5}

echo "=== DR Cross-Validation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: $CONFIG"
echo "Folds: $FOLDS"
echo "Start: $(date)"
echo "============================"

python scripts/run_cross_validation.py --config "$CONFIG" --folds "$FOLDS"

echo "Exit code: $?"
echo "End: $(date)"
