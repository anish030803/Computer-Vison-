#!/bin/bash
#SBATCH --job-name=dr-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err
#SBATCH --mail-type=END,FAIL

# NOTE: Run `module avail` on your cluster to find exact module names
module load cuda/12.x
module load python/3.11

source venv/bin/activate
mkdir -p logs/slurm

# Override via: sbatch --export=CONFIG=configs/train_dinov2.yaml slurm_train.sh
CONFIG=${CONFIG:-configs/train_efficientnet.yaml}
RESUME=${RESUME:-""}

echo "=== DR Detection Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: $CONFIG"
echo "Resume: $RESUME"
echo "Start: $(date)"
echo "=============================="

if [ -n "$RESUME" ]; then
    python scripts/run_training.py --config "$CONFIG" --resume "$RESUME"
else
    python scripts/run_training.py --config "$CONFIG"
fi

echo "Exit code: $?"
echo "End: $(date)"
