#!/bin/bash
#SBATCH --job-name=dr-preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/preprocess_%j.out
#SBATCH --error=logs/slurm/preprocess_%j.err
#SBATCH --mail-type=END,FAIL

# Northeastern Explorer cluster modules
module load python/3.13.5

source venv/bin/activate
mkdir -p logs/slurm

echo "=== DR Data Preprocessing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================="

python scripts/run_cleaning.py --config configs/data_config.yaml

echo "Exit code: $?"
echo "End: $(date)"
