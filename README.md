# Automated Diabetic Retinopathy Detection & Grading System

Deep learning system for classifying retinal fundus images into 5 DR severity grades. Trained on APTOS 2019 dataset (2,681 cleaned images) using EfficientNet-B4 and ResNet-50. Includes a training pipeline (HPC/SLURM), FastAPI inference server, and Next.js web frontend.

**Links:**
- 📊 Dataset: [APTOS 2019 on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)
- 🤗 Models: [EfficientNet-B4](https://huggingface.co/anishanish383/dr-detection-efficientnet-b4) · [ResNet-50](https://huggingface.co/anishanish383/dr-detection-resnet50)
- 📁 Source code: [GitHub](https://github.com/anish030803/Computer-Vison-)

## Results

**5-Fold Cross-Validation on EfficientNet-B4:**

| Metric | Mean ± Std | Held-Out Test Set |
|--------|-----------|-------------------|
| **QWK** | **0.8187 ± 0.0253** | **0.8076** |
| Accuracy | 0.7355 ± 0.0270 | 0.7361 |
| Macro F1 | 0.5849 ± 0.0326 | — |
| Severe DR Recall | 0.5847 ± 0.0522 | 0.5714 |
| Proliferative DR Recall | 0.4318 ± 0.0844 | 0.4091 |

**Single Train/Val Run Comparison:**

| Model | Best Val QWK | Params |
|-------|-------------|--------|
| ResNet-50 (baseline) | 0.7281 | 24M |
| **EfficientNet-B4 (primary)** | **0.7884** | 18M |

Training environment: NVIDIA A100 80GB on Northeastern Explorer HPC, BF16 mixed precision.

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
src/training/     Training loop, losses, schedulers, callbacks, metrics, CV
src/evaluation/   Evaluation suite, Grad-CAM, threshold optimization
scripts/          Entry point scripts
server/           FastAPI inference server
frontend/         Next.js web application
```

## Setup

### Prerequisites

- Python 3.11+ (tested on 3.13.5)
- CUDA 12.x (for GPU training) or Apple Silicon MPS (for local inference)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Installation

```bash
git clone https://github.com/anish030803/Computer-Vison-.git
cd Computer-Vison-

python -m venv venv
source venv/bin/activate

# CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Or CPU/MPS (Mac):
pip install torch torchvision

pip install -e .
```

### HPC Setup (SLURM — Northeastern Explorer)

```bash
module load python/3.13.5
module load cuda/12.8.0

# Submit training job
sbatch --partition=gpu --gres=gpu:a100:1 --mem=64G --time=04:00:00 \
       --output=logs/train_%j.out --wrap="source venv/bin/activate && \
       python scripts/run_training.py --config configs/train_efficientnet.yaml"

# Or interactive session
srun --partition=gpu --gres=gpu:a100:1 --mem=64G --time=04:00:00 --pty bash
```

## Pipeline

The full workflow runs through 11 steps:

### 1. Download data
```bash
kaggle datasets download -d mariaherrerot/aptos2019 -p data/raw/aptos2019/
```

### 2. Analyze raw data (recommends cleaning thresholds)
```bash
python scripts/analyze_data.py --data-dir data/raw/aptos2019 --stage raw
```

### 3-4. Clean + preprocess
```bash
python scripts/run_cleaning.py --config configs/data_config.yaml --datasets aptos2019 --skip-download
```

Runs 5-pass iterative cleaning (file integrity → duplicates → quality → resolution → labels), then Ben Graham's preprocessing to 380x380, cached as .npy files.

### 5. Hyperparameters auto-computed from cleaned data
- Class weights: inverse frequency
- Focal loss gamma: based on imbalance ratio
- Splits: 80/10/10 stratified
- Seed: 42

### 6-8. Train with two-phase strategy
```bash
python scripts/run_training.py --config configs/train_efficientnet.yaml
```

- **Phase 1**: frozen backbone, train classification head only (warmup, 20 epochs)
- **Phase 2**: unfreeze top 30% of backbone, fine-tune with lower LR (15-25 epochs)
- Linear warmup → cosine annealing scheduler
- Class-weighted cross-entropy with label smoothing
- BF16 mixed precision, gradient clipping, MixUp augmentation
- Early stopping on val_qwk (patience=5-7)

### 9. Cross-validation
```bash
python scripts/run_cross_validation.py --config configs/train_efficientnet.yaml --folds 5
```

Holds out 10% test set, runs 5-fold stratified CV on remaining 90%, reports mean ± std.

### 10. Test locally
```bash
python scripts/run_inference.py --image fundus.png \
       --checkpoint checkpoints/efficientnet_b4/best.pt
```

### 11. Web frontend (FastAPI + Next.js)
```bash
# Backend
MODEL_CONFIG=configs/train_efficientnet.yaml \
CHECKPOINT_PATH=checkpoints/efficientnet_b4/best.pt \
python -m uvicorn server.main:app --port 8000

# Frontend
cd frontend && bun install && bun dev
# Open http://localhost:3000
```

## Configuration

| File | Description |
|------|-------------|
| `configs/data_config.yaml` | Datasets, cleaning thresholds, preprocessing |
| `configs/train_efficientnet.yaml` | EfficientNet-B4 training (primary model) |
| `configs/train_dinov2.yaml` | DINOv2 ViT-L/14 training |
| `configs/train_resnet.yaml` | ResNet-50 baseline |

## Models

| Model | Input Size | Batch Size | Params | Status |
|-------|-----------|------------|--------|--------|
| **EfficientNet-B4** | 380×380 | 64 | 18M | Trained — QWK 0.82 |
| DINOv2 ViT-L/14 | 518×518 | 32 | 304M | Not yet trained |
| ResNet-50 | 380×380 | 128 | 24M | Trained — QWK 0.73 |

All models use two-phase training: head warmup (frozen backbone) then fine-tuning (top layers unfrozen).

## Dataset

**APTOS 2019 Blindness Detection** — [Kaggle dataset](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)

Download via:
```bash
kaggle datasets download -d mariaherrerot/aptos2019 -p data/raw/aptos2019/
```

- Original: 3,662 training images
- After cleaning: **2,681 images**
- Class distribution: 49% No DR, 10% Mild, 28% Moderate, 5% Severe, 8% Proliferative
- Imbalance ratio: 9.5x

Cleaning pipeline removed duplicates (pHash hamming < 3) and quality outliers using thresholds calibrated from the actual data distribution (2nd percentile cutoffs for sharpness, brightness, contrast).

## Target vs. Achieved Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| QWK | ≥ 0.85 | 0.82 | Close, not yet hit |
| Severe DR Recall | ≥ 0.80 | 0.58 | Not safe for clinical use |
| Proliferative DR Recall | ≥ 0.80 | 0.43 | Not safe for clinical use |
| Inference latency | < 2s | < 1s on A100/MPS | Met |

The model is suitable for **screening triage** but not yet for autonomous diagnosis. Improvements needed:
- More Severe + Proliferative training data (only 137 + 219 samples)
- Test-time augmentation
- Ensemble with DINOv2

## References

- Gulshan et al. (2016). Development and Validation of a Deep Learning Algorithm for Detection of DR. JAMA, 316(22), 2402–2410.
- Graham, B. (2015). Kaggle Diabetic Retinopathy Detection Competition Report. University of Warwick.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- Oquab et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 618–626.
- APTOS 2019 Blindness Detection Dataset (Kaggle).

## License

MIT
