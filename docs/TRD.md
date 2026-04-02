# Technical Requirements Document (TRD)

## Automated Diabetic Retinopathy Detection & Grading System

**Version:** 1.0
**Repository:** https://github.com/anish030803/Computer-Vison-.git
**Compute:** University HPC — NVIDIA H200 GPUs

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Frontend (Next.js)                    │
│  Upload → Predict → Grad-CAM Overlay → Dashboard → Analytics    │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────┐
│                     Inference Server (FastAPI)                    │
│  Model Loading → Preprocessing → Prediction → Grad-CAM Gen      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Trained Model Artifacts                        │
│  EfficientNet-B4 / DINOv2 / ResNet-50 Baseline                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Training Pipeline (HPC)                      │
│  Data Download → Cleaning Loop → Preprocessing → Training        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Infrastructure & Compute

### 2.1 Training Environment (University HPC)
- **GPU**: NVIDIA H200 (80GB HBM3 per GPU)
- **Framework**: PyTorch 2.x (primary), TensorFlow 2.x (baseline comparison)
- **Python**: 3.11+
- **Job Scheduler**: SLURM (assumed; adapt scripts if PBS/SGE)
- **Storage**: Shared filesystem for datasets, checkpoints, logs
- **Monitoring**: TensorBoard, Weights & Biases (optional)

### 2.2 Development Environment
- **IDE**: VS Code with Claude extension
- **Version Control**: Git → GitHub (phased pushes)
- **Containerization**: Docker for reproducible environments

### 2.3 Frontend Stack
- **Framework**: Next.js (latest, App Router)
- **Language**: TypeScript (strict mode)
- **Runtime/Bundler**: Bun
- **Styling**: Tailwind CSS
- **Components**: Animate UI, Motion library
- **Forms**: React Hook Form + Zod validation
- **Data Fetching**: React Query (TanStack Query)
- **Image Optimization**: Sharp
- **Linting**: oxlint
- **Architecture**: SOLID principles
- **State Management**: React Query cache + React context where needed

---

## 3. Data Pipeline (Detailed)

### 3.1 Data Discovery & Download

**Primary Datasets (auto-download with verification):**

| Dataset | Images | Classes | Source | Priority |
|---------|--------|---------|--------|----------|
| APTOS 2019 Blindness Detection | 3,662 (train) + 1,928 (test) | 5 (0–4 severity) | Kaggle | P0 — Primary |
| EyePACS | ~88,000 | 5 | Kaggle | P1 — Scale |
| Messidor-2 | 1,748 | 4 grades + DME | ADCIS | P1 — Cross-validation |
| IDRiD (Indian Diabetic Retinopathy) | 516 | 5 + lesion masks | IEEE DataPort | P2 — Segmentation |
| DDR (Diabetic Retinopathy Detection) | 13,673 | 6 | GitHub/Academic | P2 — Scale |

**Download Strategy:**
```
1. Check if dataset exists locally (hash verification)
2. Download via Kaggle API / direct URL / academic portal
3. Verify checksums against known hashes
4. Extract and organize into standardized directory structure
5. Log download metadata (source, date, version, hash)
```

### 3.2 Iterative Data Cleaning Loop

The cleaning pipeline runs in a loop until all quality gates pass. Each iteration logs its findings.

```
REPEAT:
  ├── Pass 1: File Integrity
  │   ├── Verify all images loadable (PIL/OpenCV)
  │   ├── Remove corrupt / truncated files
  │   ├── Log removed files with reasons
  │   └── GATE: 0 corrupt files remaining
  │
  ├── Pass 2: Duplicate Detection
  │   ├── Compute perceptual hashes (pHash)
  │   ├── Flag exact and near-duplicates (hamming distance < 5)
  │   ├── Keep highest quality version of duplicates
  │   └── GATE: 0 duplicate pairs remaining
  │
  ├── Pass 3: Quality Assessment
  │   ├── Compute image quality metrics (sharpness via Laplacian variance, brightness, contrast)
  │   ├── Flag images below quality thresholds
  │   ├── Remove or quarantine low-quality images
  │   └── GATE: All images above minimum quality threshold
  │
  ├── Pass 4: Resolution & Format Normalization
  │   ├── Verify minimum resolution (≥ 256×256 raw)
  │   ├── Convert non-standard formats to PNG/JPEG
  │   ├── Log resolution distribution statistics
  │   └── GATE: Uniform format, all above minimum resolution
  │
  ├── Pass 5: Label Verification
  │   ├── Cross-check labels against source CSV/metadata
  │   ├── Flag missing or ambiguous labels
  │   ├── Compute class distribution and log imbalance ratio
  │   └── GATE: All images have verified labels
  │
  └── VERIFY: Run comprehensive validation report
      ├── Total images per class
      ├── Quality score distribution
      ├── Image dimension statistics
      ├── Sample visualization grid (5 per class)
      └── DECISION: Pass → proceed | Fail → re-run with adjusted thresholds
UNTIL all gates pass
```

### 3.3 Preprocessing Pipeline (Ben Graham's Method)

```python
def ben_graham_preprocess(image, target_size=380):
    """
    1. Resize to target_size × target_size
    2. Apply circular crop (remove black borders)
    3. Local average color subtraction (Gaussian blur, sigma ~ 10% of image width)
    4. Add back 128 to maintain contrast
    5. Apply circular mask
    6. Normalize pixel values to [0, 1]
    """
    # Exact implementation in src/data/preprocessing.py
```

**Additional Preprocessing Options:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced microaneurysm visibility
- Green channel extraction (highest contrast for retinal lesions)
- Optional: multi-scale preprocessing at 224, 380, 512 for ensemble

### 3.4 Dynamic Hyperparameter Configuration

After data cleaning, automatically compute and set:

```yaml
# Auto-computed from cleaned data statistics
data_config:
  num_classes: 5
  class_weights: [computed from inverse frequency]
  class_distribution: [computed]
  total_train_images: [counted]
  total_val_images: [counted]
  total_test_images: [counted]
  mean_image_quality: [computed]
  
training_config:
  batch_size: [auto: based on GPU memory and image size]
  initial_lr: [auto: scaled by batch size — linear scaling rule]
  warmup_epochs: [auto: ~5-10% of total epochs]
  total_epochs_phase1: 20
  total_epochs_phase2: 15
  mixup_alpha: 0.2  # higher if severe imbalance
  label_smoothing: 0.1
  focal_loss_gamma: [auto: higher if extreme imbalance detected]
```

---

## 4. Model Architectures

### 4.1 EfficientNet-B4 (Primary Model)

**Architecture:**
```
Input (380×380×3)
    │
    ▼
EfficientNet-B4 Backbone (ImageNet pretrained)
    │ Compound scaling: depth=1.8, width=1.4, resolution=380
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
BatchNormalization
    │
    ▼
Dropout (p=0.4)
    │
    ▼
Dense (256, ReLU) + L2 regularization
    │
    ▼
Dropout (p=0.3)
    │
    ▼
Dense (5, Softmax) → [No DR, Mild, Moderate, Severe, Proliferative]
```

**Training Strategy:**
```
Phase 1 — Head Warmup (20 epochs max):
  ├── Backbone: FROZEN
  ├── Trainable: Classification head only
  ├── Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  ├── LR Schedule: Linear warmup (5 epochs) → Cosine annealing
  ├── Loss: Weighted categorical cross-entropy + label smoothing (0.1)
  └── Purpose: Stabilize head weights before backbone modification

Phase 2 — Fine-tuning (15+ epochs):
  ├── Backbone: Top 30% layers UNFROZEN
  ├── Trainable: Head + top backbone layers
  ├── Optimizer: AdamW (lr=1e-5, weight_decay=1e-4)
  ├── LR Schedule: Cosine annealing with warm restarts
  ├── Loss: Same as Phase 1 (optionally switch to focal loss)
  └── Purpose: Adapt backbone features to fundus domain
```

### 4.2 DINOv2 (Alternative/Ensemble Model)

DINOv2 is a self-supervised Vision Transformer pretrained on 142M curated images. It provides high-quality visual features without task-specific labels, making it ideal for medical imaging where labeled data is scarce.

**Classification Head:**
```
Input (518×518×3)  # DINOv2 ViT-L native resolution
    │
    ▼
DINOv2 ViT-L/14 Backbone (frozen or partially unfrozen)
    │ [CLS] token output: 1024-dim feature vector
    │
    ▼
LayerNorm
    │
    ▼
Linear (1024 → 512, GELU)
    │
    ▼
Dropout (p=0.3)
    │
    ▼
Linear (512 → 5, Softmax)
```

**Optional Segmentation Head (for lesion localization):**
```
DINOv2 patch tokens (spatial features)
    │
    ▼
Linear projection to segmentation space
    │
    ▼
Upsampling decoder (simple FPN or linear)
    │
    ▼
Per-pixel lesion class predictions
```

**Transfer Learning with DINOv2:**
- Phase 1: Freeze DINOv2 entirely, train linear head (probe)
- Phase 2: Unfreeze last 4–6 transformer blocks, fine-tune with low LR
- Advantage: Self-supervised features capture general visual patterns; particularly good for rare pathological features not well-represented in ImageNet

### 4.3 ResNet-50 (Baseline)

Standard ResNet-50 with ImageNet weights, same classification head structure, same training pipeline. Used purely as a performance reference point.

### 4.4 Ensemble Strategy (Optional)

If individual models plateau below QWK 0.85:
```
Final Prediction = weighted_average(
    EfficientNet-B4_probs × w1,
    DINOv2_probs × w2,
    ResNet-50_probs × w3
)
# Weights optimized on validation set
# Test-Time Augmentation (TTA): average predictions over 5 augmented versions
```

---

## 5. Overfitting Prevention Strategy

Multiple complementary mechanisms applied throughout training:

| Technique | Configuration | Phase |
|-----------|--------------|-------|
| Early Stopping | patience=5, monitor=val_qwk | Both |
| ReduceLROnPlateau | factor=0.5, patience=3 | Both |
| Dropout | 0.4 (post-GAP), 0.3 (post-Dense) | Both |
| Label Smoothing | 0.1 | Both |
| Weight Decay (L2) | 1e-4 via AdamW | Both |
| Data Augmentation | Geometric + color (see §3.4 of PRD) | Both |
| MixUp Augmentation | alpha=0.2 | Phase 2 |
| Gradient Clipping | max_norm=1.0 | Phase 2 |
| Conservative Unfreezing | Only top 30% of backbone | Phase 2 |
| Stochastic Depth | p=0.2 (if using EfficientNet) | Both |
| Checkpoint Selection | Best val_qwk, not last epoch | Both |

---

## 6. Evaluation Framework

### 6.1 Metrics

**Primary:**
- Quadratic Weighted Kappa (QWK) — ordinal agreement metric
- Per-class Recall (especially Severe + Proliferative DR)

**Secondary:**
- Overall Accuracy
- Per-class Precision, Recall, F1-score
- AUC-ROC (per-class, one-vs-rest)
- AUC-PR (per-class, especially minority classes)
- Normalized Confusion Matrix
- Cohen's Kappa (unweighted, for comparison)

### 6.2 Explainability

**Grad-CAM Visualizations:**
- Generate for all test set predictions
- Highlight retinal regions driving classification decisions
- Curated examples across all 5 severity grades for the final report
- Sanity check: verify model attends to clinically relevant regions (lesions, not artifacts)

**Prediction Threshold Optimization:**
- After training, sweep thresholds to maximize sensitivity for Severe/Proliferative DR
- Report operating point trade-offs (sensitivity vs specificity curves)
- Clinical deployment would favor higher sensitivity even at cost of more false positives

### 6.3 Cross-Dataset Validation

- Train on APTOS 2019 → evaluate on Messidor-2 subset
- Report domain shift metrics and performance degradation
- Document camera/population differences

---

## 7. Project Directory Structure

```
Computer-Vision/
├── CLAUDE.md                    # Instructions for Claude Code extension
├── README.md                    # Project overview and setup
├── .gitignore
├── pyproject.toml               # Python dependencies (uv/pip)
├── bun.lockb                    # Frontend lockfile
├── docker-compose.yml
│
├── docs/
│   ├── PRD.md                   # Product Requirements Document
│   ├── TRD.md                   # Technical Requirements Document
│   └── research/                # Literature notes, peer feedback summaries
│
├── tasks/
│   ├── todo.md                  # Phase-based task tracking
│   └── lessons.md               # Self-improvement log
│
├── configs/
│   ├── data_config.yaml         # Dataset paths, cleaning thresholds
│   ├── train_efficientnet.yaml  # EfficientNet-B4 training config
│   ├── train_dinov2.yaml        # DINOv2 training config
│   ├── train_resnet.yaml        # ResNet-50 baseline config
│   └── hpc/
│       ├── slurm_train.sh       # SLURM job script for training
│       └── slurm_preprocess.sh  # SLURM job script for data prep
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # Auto-discovery and download
│   │   ├── cleaning.py          # Iterative cleaning loop
│   │   ├── preprocessing.py     # Ben Graham's pipeline + CLAHE
│   │   ├── augmentation.py      # Training augmentations
│   │   ├── dataset.py           # PyTorch Dataset / DataLoader
│   │   └── validation.py        # Data quality verification
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── efficientnet.py      # EfficientNet-B4 + custom head
│   │   ├── dinov2.py            # DINOv2 classification + segmentation
│   │   ├── resnet_baseline.py   # ResNet-50 baseline
│   │   ├── ensemble.py          # Model ensemble logic
│   │   └── heads.py             # Shared classification/segmentation heads
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Main training loop (two-phase)
│   │   ├── losses.py            # Weighted CE, focal loss, mixup
│   │   ├── schedulers.py        # LR warmup + cosine annealing
│   │   ├── callbacks.py         # Early stopping, checkpointing, logging
│   │   └── metrics.py           # QWK, per-class metrics, AUC
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py          # Full evaluation suite
│   │   ├── gradcam.py           # Grad-CAM generation
│   │   ├── threshold_opt.py     # Prediction threshold optimization
│   │   └── cross_dataset.py     # Domain shift evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML config loader
│       ├── logging.py           # Structured logging
│       ├── seed.py              # Reproducibility (seed everything)
│       └── checkpoint.py        # Save/load model checkpoints
│
├── scripts/
│   ├── run_cleaning.py          # Entry: data download + cleaning loop
│   ├── run_training.py          # Entry: full training pipeline
│   ├── run_evaluation.py        # Entry: evaluation suite
│   └── run_inference.py         # Entry: single image prediction
│
├── frontend/
│   ├── package.json
│   ├── bun.lockb
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx             # Landing / upload page
│   │   ├── predict/
│   │   │   └── page.tsx         # Prediction results + Grad-CAM
│   │   ├── dashboard/
│   │   │   └── page.tsx         # Analytics dashboard
│   │   └── api/
│   │       └── predict/
│   │           └── route.ts     # API route proxying to inference server
│   ├── components/
│   │   ├── ui/                  # Animate UI + custom components
│   │   ├── upload/
│   │   ├── prediction/
│   │   └── dashboard/
│   ├── lib/
│   │   ├── api.ts               # React Query hooks
│   │   ├── schemas.ts           # Zod validation schemas
│   │   └── utils.ts
│   └── public/
│
├── server/
│   ├── main.py                  # FastAPI inference server
│   ├── model_loader.py
│   └── schemas.py               # Pydantic request/response models
│
├── tests/
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
│
└── notebooks/
    ├── 01_eda.ipynb             # Exploratory data analysis
    ├── 02_preprocessing_viz.ipynb
    ├── 03_training_analysis.ipynb
    └── 04_gradcam_analysis.ipynb
```

---

## 8. HPC Configuration

### 8.1 SLURM Job Template

```bash
#!/bin/bash
#SBATCH --job-name=dr-detection-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load cuda/12.x
module load python/3.11

source venv/bin/activate
python scripts/run_training.py --config configs/train_efficientnet.yaml
```

### 8.2 GPU Memory Budget (H200 — 80GB)

With H200 GPUs, we can afford generous batch sizes and full-resolution images:

| Model | Image Size | Batch Size | Est. GPU Memory |
|-------|-----------|------------|-----------------|
| EfficientNet-B4 | 380×380 | 64 | ~25 GB |
| DINOv2 ViT-L/14 | 518×518 | 32 | ~40 GB |
| ResNet-50 | 380×380 | 128 | ~20 GB |

Mixed precision (FP16/BF16) enabled by default on H200 for ~2x throughput.

---

## 9. API Specification

### POST /api/predict

**Request:**
```json
{
  "image": "<base64-encoded fundus image>",
  "return_gradcam": true
}
```

**Response:**
```json
{
  "prediction": {
    "class": 2,
    "label": "Moderate NPDR",
    "confidence": 0.78,
    "probabilities": {
      "No DR": 0.05,
      "Mild NPDR": 0.12,
      "Moderate NPDR": 0.78,
      "Severe NPDR": 0.04,
      "Proliferative DR": 0.01
    }
  },
  "gradcam": {
    "heatmap": "<base64-encoded heatmap overlay>",
    "attention_regions": ["temporal_retina", "macula"]
  },
  "metadata": {
    "model": "efficientnet-b4-v1.2",
    "inference_time_ms": 142,
    "preprocessing_applied": "ben_graham_380"
  }
}
```

---

## 10. Dependencies

### Python (Training)
```
torch>=2.2
torchvision>=0.17
timm>=1.0               # EfficientNet-B4, DINOv2
albumentations>=1.4      # Advanced augmentation
opencv-python>=4.9
scikit-learn>=1.4
pandas>=2.2
numpy>=1.26
pillow>=10.0
matplotlib>=3.8
seaborn>=0.13
tensorboard>=2.16
wandb>=0.16              # Optional
pyyaml>=6.0
tqdm>=4.66
imagehash>=4.3           # Duplicate detection
grad-cam>=1.5
kaggle>=1.6              # Dataset download
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "next": "latest",
    "react": "latest",
    "react-dom": "latest",
    "@tanstack/react-query": "latest",
    "react-hook-form": "latest",
    "zod": "latest",
    "@hookform/resolvers": "latest",
    "motion": "latest",
    "sharp": "latest",
    "tailwindcss": "latest",
    "animate-ui": "latest"
  },
  "devDependencies": {
    "typescript": "latest",
    "oxlint": "latest",
    "@types/react": "latest",
    "@types/node": "latest"
  }
}
```
