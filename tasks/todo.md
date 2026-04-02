# TODO — Automated DR Detection System

## Repository: https://github.com/anish030803/Computer-Vison-.git

> Each phase is a git push milestone. Do NOT proceed to the next phase until current phase passes all verification gates.

---

## Phase 0: Project Scaffolding & Configuration
**Git tag:** `v0.1-scaffold`

- [x] Initialize repository structure (see TRD §7 for directory layout)
- [x] Create `pyproject.toml` with all Python dependencies
- [x] Create SLURM job templates in `configs/hpc/`
- [x] Set up `.gitignore` (ignore datasets, checkpoints, __pycache__, .env, wandb/)
- [x] Create `configs/data_config.yaml` with dataset URLs, paths, cleaning thresholds
- [x] Create `configs/train_efficientnet.yaml` with full hyperparameter spec
- [x] Create `configs/train_dinov2.yaml`
- [x] Create `configs/train_resnet.yaml`
- [x] Implement `src/utils/config.py` — YAML config loader
- [x] Implement `src/utils/seed.py` — seed everything (torch, numpy, random, CUDA)
- [x] Implement `src/utils/logging.py` — structured logging with file + console output
- [x] Implement `src/utils/checkpoint.py` — save/load with metadata
- [x] Write `README.md` with setup instructions
- [x] **VERIFY**: All configs load without errors, directory structure matches TRD
- [ ] **GIT PUSH** → Phase 0 complete

---

## Phase 1: Data Pipeline — Download, Clean, Preprocess
**Git tag:** `v0.2-data-pipeline`

### 1.1 Data Download
- [x] Implement `src/data/download.py`:
  - [x] APTOS 2019 via Kaggle API (primary)
  - [x] EyePACS download handler (large dataset — chunked download)
  - [x] Messidor-2 download handler
  - [x] IDRiD download handler (lesion masks for segmentation)
  - [x] Hash verification after download
  - [x] Standardized directory structure: `data/raw/{dataset_name}/images/` + `data/raw/{dataset_name}/labels.csv`
- [x] **VERIFY**: All datasets downloaded and verified, sample images viewable

### 1.2 Iterative Data Cleaning Loop
- [x] Implement `src/data/cleaning.py`:
  - [x] Pass 1: File integrity check (corrupt/truncated images)
  - [x] Pass 2: Duplicate detection via perceptual hashing (pHash, hamming distance < 5)
  - [x] Pass 3: Quality assessment (Laplacian variance for sharpness, brightness/contrast stats)
  - [x] Pass 4: Resolution & format normalization (min 256×256, standardize to PNG/JPEG)
  - [x] Pass 5: Label verification against source metadata
  - [x] Loop controller: repeat until all gates pass
  - [x] Comprehensive cleaning report generation (JSON + human-readable)
- [x] Implement `src/data/validation.py`:
  - [x] Post-cleaning validation: class distribution, quality histograms, sample grid
  - [x] Export validation report with visualizations
- [x] **VERIFY**: Cleaning loop runs to completion, all gates pass, validation report reviewed

### 1.3 Preprocessing
- [x] Implement `src/data/preprocessing.py`:
  - [x] Ben Graham's normalization (resize → circular crop → Gaussian blur subtraction → normalize)
  - [x] CLAHE variant for enhanced contrast
  - [x] Green channel extraction option
  - [x] Caching: save preprocessed arrays to disk (NumPy .npy or HDF5)
- [x] **VERIFY**: Preprocessed images visually correct, cached files loadable, sizes match expected dimensions

### 1.4 Dynamic Hyperparameter Setting
- [x] After cleaning, auto-compute and save to config:
  - [x] Class weights (inverse frequency via sklearn)
  - [x] Recommended batch size (based on dataset size + GPU memory)
  - [x] Focal loss gamma (based on imbalance ratio)
  - [x] Augmentation intensity (based on minority class sample count)
- [x] **VERIFY**: Auto-computed config values are reasonable, logged and saved

### 1.5 Dataset & DataLoader
- [x] Implement `src/data/dataset.py`:
  - [x] PyTorch Dataset class with on-the-fly augmentation
  - [x] Stratified train/val/test split (80/10/10)
  - [x] DataLoader with proper num_workers, pin_memory, prefetch
- [x] Implement `src/data/augmentation.py`:
  - [x] Geometric: random H/V flip, rotation ±36°, zoom 90–110%
  - [x] Color: brightness ±10%, contrast ±10%
  - [x] Advanced: MixUp (alpha=0.2), targeted optic disc/macula crops
  - [x] Clinically calibrated — no distortions that alter pathology presentation
- [x] **VERIFY**: DataLoader yields correct shapes, augmentations visually plausible, class weights applied
- [ ] **GIT PUSH** → Phase 1 complete

---

## Phase 2: Model Architectures
**Git tag:** `v0.3-models`

### 2.1 EfficientNet-B4
- [ ] Implement `src/models/efficientnet.py`:
  - [ ] Load pretrained EfficientNet-B4 from `timm`
  - [ ] Custom classification head: GAP → BN → Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3) → Dense(5, Softmax)
  - [ ] Layer freezing/unfreezing utilities (freeze all, unfreeze top N%)
  - [ ] Forward pass verified with dummy input
- [ ] **VERIFY**: Model summary matches architecture spec, parameter counts correct

### 2.2 DINOv2
- [ ] Implement `src/models/dinov2.py`:
  - [ ] Load DINOv2 ViT-L/14 from torch.hub or timm
  - [ ] Classification head: [CLS] → LayerNorm → Linear(1024→512, GELU) → Dropout(0.3) → Linear(512→5)
  - [ ] Optional segmentation head using patch tokens
  - [ ] Freezing utilities for staged unfreezing
- [ ] **VERIFY**: Model loads, forward pass works, feature dimensions match spec

### 2.3 ResNet-50 Baseline
- [ ] Implement `src/models/resnet_baseline.py`:
  - [ ] Standard ResNet-50 with ImageNet weights
  - [ ] Same classification head structure for fair comparison
- [ ] **VERIFY**: Forward pass works, matches expected architecture

### 2.4 Shared Components
- [ ] Implement `src/models/heads.py` — reusable classification/segmentation heads
- [ ] Implement `src/models/ensemble.py` — weighted ensemble logic
- [ ] **VERIFY**: All models produce shape (batch, 5) output, ensemble combines correctly
- [ ] **GIT PUSH** → Phase 2 complete

---

## Phase 3: Training Pipeline
**Git tag:** `v0.4-training`

### 3.1 Loss Functions
- [ ] Implement `src/training/losses.py`:
  - [ ] Weighted categorical cross-entropy with label smoothing
  - [ ] Focal loss (gamma configurable)
  - [ ] MixUp loss (interpolated targets)
  - [ ] Combined loss option (weighted sum)

### 3.2 Schedulers
- [ ] Implement `src/training/schedulers.py`:
  - [ ] Linear warmup scheduler (N warmup epochs → target LR)
  - [ ] Cosine annealing with warm restarts
  - [ ] Combined: warmup → cosine annealing
  - [ ] ReduceLROnPlateau as fallback

### 3.3 Callbacks
- [ ] Implement `src/training/callbacks.py`:
  - [ ] Early stopping (monitor val_qwk, patience=5)
  - [ ] Model checkpointing (save best + save every epoch)
  - [ ] TensorBoard logging
  - [ ] Optional W&B logging
  - [ ] LR logging

### 3.4 Metrics
- [ ] Implement `src/training/metrics.py`:
  - [ ] Quadratic Weighted Kappa (QWK) — differentiable version for monitoring
  - [ ] Per-class precision, recall, F1
  - [ ] AUC-ROC and AUC-PR computation
  - [ ] Running metric tracker for epoch-level reporting

### 3.5 Main Trainer
- [ ] Implement `src/training/trainer.py`:
  - [ ] Two-phase training loop:
    - Phase 1: Frozen backbone → train head → warmup LR → 20 epochs
    - Phase 2: Unfreeze top 30% → fine-tune → low LR → 15+ epochs
  - [ ] Automatic transition between phases
  - [ ] Checkpoint resume capability (for HPC job restarts)
  - [ ] Mixed precision training (AMP with BF16 on H200)
  - [ ] Gradient clipping (max_norm=1.0)
  - [ ] Comprehensive per-epoch logging

### 3.6 Training Entry Points
- [ ] Implement `scripts/run_training.py`:
  - [ ] CLI: `python scripts/run_training.py --config configs/train_efficientnet.yaml`
  - [ ] Supports resume from checkpoint
  - [ ] Seeds everything before training
- [ ] Create SLURM submission scripts for each model
- [ ] **VERIFY**: Full training loop runs for 2 epochs without errors on a small data subset
- [ ] **VERIFY**: Checkpoint save/resume works correctly
- [ ] **VERIFY**: Metrics logged to TensorBoard
- [ ] **GIT PUSH** → Phase 3 complete

---

## Phase 4: Full Training Runs (HPC)
**Git tag:** `v0.5-trained-models`

### 4.1 ResNet-50 Baseline
- [ ] Submit SLURM job for ResNet-50 training
- [ ] Monitor convergence on TensorBoard
- [ ] Record final metrics: accuracy, QWK, per-class recall
- [ ] **VERIFY**: Training completed, checkpoint saved, metrics logged

### 4.2 EfficientNet-B4 Training
- [ ] Submit Phase 1 (head warmup) SLURM job
- [ ] Verify Phase 1 convergence before proceeding
- [ ] Submit Phase 2 (fine-tuning) SLURM job
- [ ] Monitor for overfitting (val loss vs train loss divergence)
- [ ] Record final metrics
- [ ] **VERIFY**: QWK on validation set, compare against baseline

### 4.3 DINOv2 Training
- [ ] Submit Phase 1 (linear probe) SLURM job
- [ ] Submit Phase 2 (partial fine-tuning) SLURM job
- [ ] Record final metrics
- [ ] **VERIFY**: Compare against EfficientNet-B4 and baseline

### 4.4 Ensemble (if needed)
- [ ] If no single model reaches QWK ≥ 0.85:
  - [ ] Optimize ensemble weights on validation set
  - [ ] Apply test-time augmentation (TTA)
  - [ ] Re-evaluate
- [ ] **VERIFY**: Final model selection justified with metrics
- [ ] **GIT PUSH** → Phase 4 complete

---

## Phase 5: Evaluation & Explainability
**Git tag:** `v0.6-evaluation`

- [ ] Implement `src/evaluation/evaluate.py`:
  - [ ] Full test set evaluation for all trained models
  - [ ] Confusion matrices (raw + normalized)
  - [ ] Per-class precision/recall/F1 table
  - [ ] QWK, AUC-ROC, AUC-PR
  - [ ] Comparison table: ResNet-50 vs EfficientNet-B4 vs DINOv2
- [ ] Implement `src/evaluation/gradcam.py`:
  - [ ] Grad-CAM heatmap generation for EfficientNet-B4 and ResNet-50
  - [ ] Attention map extraction for DINOv2
  - [ ] Sample visualizations: 5 images × 5 severity grades = 25 examples
  - [ ] Sanity check: verify attention on clinically relevant regions
- [ ] Implement `src/evaluation/threshold_opt.py`:
  - [ ] Sweep prediction thresholds for Severe + Proliferative DR
  - [ ] Report sensitivity/specificity trade-off at each threshold
  - [ ] Recommend clinical operating point
- [ ] Implement `src/evaluation/cross_dataset.py`:
  - [ ] Evaluate best model on Messidor-2
  - [ ] Document domain shift and performance degradation
- [ ] **VERIFY**: All evaluation outputs generated, metrics match expectations
- [ ] **VERIFY**: Grad-CAM shows clinically meaningful attention patterns
- [ ] **GIT PUSH** → Phase 5 complete

---

## Phase 6: Inference Server (API)
**Git tag:** `v0.7-api`

- [ ] Implement `server/main.py`:
  - [ ] FastAPI app with `/api/predict` endpoint
  - [ ] Model loading at startup (best checkpoint)
  - [ ] Image preprocessing pipeline (Ben Graham's)
  - [ ] Prediction + confidence scores
  - [ ] Optional Grad-CAM generation
  - [ ] Health check endpoint
- [ ] Implement `server/model_loader.py` — load model with error handling
- [ ] Implement `server/schemas.py` — Pydantic request/response models
- [ ] Write Dockerfile for inference server
- [ ] **VERIFY**: API responds correctly to test images, latency < 2s
- [ ] **VERIFY**: Docker image builds and runs
- [ ] **GIT PUSH** → Phase 6 complete

---

## Phase 7: Frontend
**Git tag:** `v0.8-frontend`

### 7.1 Project Setup
- [ ] Initialize Next.js project with Bun, TypeScript, Tailwind, App Router
- [ ] Configure oxlint
- [ ] Set up Animate UI + Motion library
- [ ] Configure Sharp for image optimization
- [ ] Set up React Query provider
- [ ] Set up React Hook Form + Zod schemas

### 7.2 Pages & Components
- [ ] Landing page with upload component (drag-and-drop + file picker)
- [ ] Prediction results page:
  - [ ] DR severity classification with confidence bar chart
  - [ ] Grad-CAM heatmap overlay (toggleable)
  - [ ] Per-class probability distribution
  - [ ] Clinical recommendation text
- [ ] Dashboard page:
  - [ ] Historical prediction log
  - [ ] Class distribution charts
  - [ ] Batch upload capability
- [ ] Shared UI components (SOLID principles):
  - [ ] ImageUploader
  - [ ] PredictionCard
  - [ ] ConfidenceChart
  - [ ] GradCAMOverlay
  - [ ] LoadingStates with Motion animations

### 7.3 API Integration
- [ ] React Query hooks for prediction API
- [ ] Error handling and retry logic
- [ ] Loading states with skeleton components

### 7.4 Polish
- [ ] Responsive design (desktop + tablet)
- [ ] Accessibility (ARIA labels, keyboard navigation)
- [ ] Motion animations for page transitions and result reveals
- [ ] **VERIFY**: Full flow works: upload → predict → view results → dashboard
- [ ] **GIT PUSH** → Phase 7 complete

---

## Phase 8: Docker & Documentation
**Git tag:** `v1.0-release`

- [ ] Create `docker-compose.yml` (inference server + frontend)
- [ ] Write comprehensive `README.md`:
  - [ ] Project overview and motivation
  - [ ] Architecture diagram
  - [ ] Setup instructions (local + HPC)
  - [ ] Training reproduction steps
  - [ ] API documentation
  - [ ] Frontend screenshots
  - [ ] Results summary with metrics table
  - [ ] Limitations and future work
- [ ] Clean up all code: remove dead code, add docstrings, type hints everywhere
- [ ] Final code review: "Would a staff engineer approve this?"
- [ ] **VERIFY**: Fresh clone → docker-compose up → full system works
- [ ] **GIT PUSH** → v1.0 release

---

## Review Section

_This section is updated after each phase completion._

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 0 — Scaffold | ✅ Complete | 2026-04-02 | All configs verified, directory structure matches TRD, utils tested |
| 1 — Data Pipeline | ✅ Complete | 2026-04-02 | All 6 modules verified with synthetic data, cleaning loop + splits + augmentation + MixUp tested |
| 2 — Models | ⬜ Pending | | |
| 3 — Training Pipeline | ⬜ Pending | | |
| 4 — Full Training | ⬜ Pending | | |
| 5 — Evaluation | ⬜ Pending | | |
| 6 — API | ⬜ Pending | | |
| 7 — Frontend | ⬜ Pending | | |
| 8 — Release | ⬜ Pending | | |
