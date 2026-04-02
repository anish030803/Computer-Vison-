# Automated Diabetic Retinopathy Detection & Grading System

Deep learning system for classifying retinal fundus images into 5 DR severity grades using EfficientNet-B4, DINOv2, and ResNet-50. Includes a training pipeline (HPC/SLURM), FastAPI inference server, and Next.js web frontend.

## Architecture

```
Web Frontend (Next.js) → REST API → Inference Server (FastAPI) → Trained Models
                                                                        ↑
                                          Training Pipeline (HPC) ──────┘
                                          Data → Clean → Preprocess → Train
```

## Severity Grades

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No visible retinopathy |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | More than just microaneurysms |
| 3 | Severe NPDR | Extensive intraretinal hemorrhages |
| 4 | Proliferative DR | Neovascularization or vitreous hemorrhage |

## Project Structure

```
configs/          YAML configs (data, training, HPC)
src/data/         Data pipeline (download, clean, preprocess, augment)
src/models/       Model architectures (EfficientNet, DINOv2, ResNet, ensemble)
src/training/     Training loop, losses, schedulers, callbacks, metrics
src/evaluation/   Evaluation suite, Grad-CAM, threshold optimization
scripts/          Entry point scripts
server/           FastAPI inference server
frontend/         Next.js web application
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA 12.x (for GPU training)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Installation

```bash
git clone https://github.com/anish030803/Computer-Vison-.git
cd Computer-Vision

python -m venv venv
source venv/bin/activate

# Install with CUDA support (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Optional: Weights & Biases + dev tools
pip install -e ".[wandb,dev]"
```

### HPC Setup (SLURM)

```bash
# Training (GPU)
sbatch configs/hpc/slurm_train.sh

# With specific config
sbatch --export=CONFIG=configs/train_dinov2.yaml configs/hpc/slurm_train.sh

# Resume from checkpoint
sbatch --export=CONFIG=configs/train_efficientnet.yaml,RESUME=checkpoints/efficientnet_b4/latest.pt configs/hpc/slurm_train.sh

# Data preprocessing (CPU)
sbatch configs/hpc/slurm_preprocess.sh
```

## Configuration

| File | Description |
|------|-------------|
| `configs/data_config.yaml` | Datasets, cleaning thresholds, preprocessing |
| `configs/train_efficientnet.yaml` | EfficientNet-B4 training (primary model) |
| `configs/train_dinov2.yaml` | DINOv2 ViT-L/14 training |
| `configs/train_resnet.yaml` | ResNet-50 baseline |

## Models

| Model | Input Size | Batch Size | Params | Notes |
|-------|-----------|------------|--------|-------|
| EfficientNet-B4 | 380×380 | 64 | ~19M | Primary model, compound scaling |
| DINOv2 ViT-L/14 | 518×518 | 32 | ~304M | Self-supervised, strong features |
| ResNet-50 | 380×380 | 128 | ~25M | Baseline comparison |

All models use two-phase training: head warmup (frozen backbone) then fine-tuning (top layers unfrozen).

## Usage

```bash
# Data pipeline
python scripts/run_cleaning.py --config configs/data_config.yaml

# Training
python scripts/run_training.py --config configs/train_efficientnet.yaml

# Evaluation
python scripts/run_evaluation.py --config configs/train_efficientnet.yaml

# Single image inference
python scripts/run_inference.py --image path/to/fundus.jpg --checkpoint checkpoints/efficientnet_b4/best.pt
```

## Results

| Model | QWK | Accuracy | Severe DR Recall | Proliferative DR Recall |
|-------|-----|----------|-----------------|------------------------|
| ResNet-50 | — | — | — | — |
| EfficientNet-B4 | — | — | — | — |
| DINOv2 ViT-L/14 | — | — | — | — |

*Results will be populated after Phase 4 training runs.*

## Target Metrics

- **QWK ≥ 0.85** (primary)
- **Per-class recall ≥ 0.80** for Severe + Proliferative DR
- **Inference latency < 2s** per image

## References

- Gulshan et al. (2016). JAMA, 316(22), 2402–2410.
- Graham, B. (2015). Kaggle DR Competition Report. University of Warwick.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling. ICML.
- Oquab et al. (2024). DINOv2: Learning Robust Visual Features. TMLR.
- Selvaraju et al. (2017). Grad-CAM. ICCV, 618–626.

## License

MIT
