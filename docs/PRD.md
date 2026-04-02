# Product Requirements Document (PRD)

## Automated Diabetic Retinopathy Detection & Grading System

**Version:** 1.0
**Author:** Anish
**Repository:** https://github.com/anish030803/Computer-Vison-.git
**Last Updated:** April 2026

---

## 1. Executive Summary

Build an end-to-end automated system for detecting and grading diabetic retinopathy (DR) severity from retinal fundus images. The system classifies images into five internationally recognized severity categories — No DR, Mild NPDR, Moderate NPDR, Severe NPDR, and Proliferative DR — following the International Clinical Diabetic Retinopathy Disease Severity Scale.

The system combines modern deep learning architectures (EfficientNet-B4, DINOv2) with a production-grade web frontend for clinical accessibility, targeting a Quadratic Weighted Kappa (QWK) ≥ 0.85 on held-out test data.

---

## 2. Problem Statement

Diabetic retinopathy affects approximately 93 million people worldwide (Gulshan et al., 2016) and is one of the leading causes of preventable blindness in working-age adults. Timely screening can prevent up to 90% of DR-related vision loss, but there is a severe shortage of trained ophthalmologists, especially in rural and underserved areas.

Current screening bottlenecks:
- Manual review by ophthalmologists is expensive and time-consuming
- 60–70% of screened patients have no retinopathy, wasting specialist time
- Severe cases (5–10% of patients) that urgently need referral can be missed in high-volume settings
- Rural and low-income areas lack specialist access entirely

---

## 3. Target Users

1. **Primary care clinics** without ophthalmologists — automated screening allows non-specialists to capture fundus images and get AI-driven preliminary diagnoses
2. **Public health organizations** running population-level diabetes screening programs
3. **Ophthalmologists** seeking triage assistance to prioritize severe cases
4. **Researchers** evaluating DR detection methodologies

---

## 4. Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Quadratic Weighted Kappa (QWK) | ≥ 0.85 | Clinical screening threshold; penalizes predictions far from true label |
| Per-class Recall (Severe + Proliferative DR) | ≥ 0.80 | Missing severe cases is clinically dangerous |
| Overall Accuracy | ≥ 0.82 | Secondary metric; can be misleading with class imbalance |
| AUC-ROC (per-class) | ≥ 0.90 | Discrimination ability across severity levels |
| AUC-PR (minority classes) | Track and report | More informative than ROC AUC for imbalanced classes (Nada Moursi suggestion) |
| Inference Latency | < 2 seconds per image | Practical for clinical deployment |

---

## 5. Functional Requirements

### 5.1 Data Pipeline
- **FR-01**: Automatically discover, download, and validate the best publicly available DR datasets (APTOS 2019, EyePACS, Messidor-2, IDRiD, DDR)
- **FR-02**: Implement an iterative data cleaning loop with automated quality checks (corrupt images, duplicates, resolution thresholds, class distribution verification)
- **FR-03**: Verify data integrity at each cleaning pass before proceeding
- **FR-04**: Apply Ben Graham's preprocessing pipeline: resize to 380×380 (EfficientNet-B4 native), Gaussian blur illumination normalization, circular crop to remove black borders, pixel normalization to [0, 1]
- **FR-05**: Cache preprocessed images as NumPy arrays or TFRecords for training efficiency
- **FR-06**: Compute and apply inverse-frequency class weights using scikit-learn's `compute_class_weight`
- **FR-07**: Stratified 80/10/10 train/validation/test splits preserving class proportions

### 5.2 Model Training
- **FR-08**: Implement EfficientNet-B4 backbone with custom classification head (GAP → BatchNorm → Dropout 0.4 → Dense 256 ReLU → Dropout 0.3 → Softmax 5)
- **FR-09**: Implement DINOv2 (ViT-based self-supervised model) as an alternative/ensemble backbone for classification and optional lesion segmentation
- **FR-10**: Two-phase training strategy:
  - Phase 1 (Warmup): Freeze backbone, train classification head only with higher LR (1e-3) for 20 epochs
  - Phase 2 (Fine-tuning): Unfreeze top 30% of backbone layers, fine-tune entire network with lower LR (1e-5) for 15+ epochs
- **FR-11**: Learning rate warmup schedule during first N epochs before main training
- **FR-12**: Overfitting prevention: early stopping (patience=5), ReduceLROnPlateau (factor=0.5), dropout, label smoothing (0.1), weight decay
- **FR-13**: Class imbalance mitigation: weighted cross-entropy loss, targeted augmentation for minority classes, focal loss option, MixUp augmentation
- **FR-14**: Data augmentation: random horizontal/vertical flip, rotation ±36°, zoom 90–110%, brightness ±10%, contrast ±10%, targeted crops centered on optic disc/macula region
- **FR-15**: Train ResNet-50 baseline for comparative evaluation
- **FR-16**: Set hyperparameters dynamically based on cleaned data statistics (class distribution, image quality metrics, dataset size)
- **FR-17**: Implement Grad-CAM visualizations for model interpretability and clinician trust
- **FR-18**: Generate comprehensive evaluation: confusion matrices, per-class precision/recall/F1, QWK, AUC-ROC, AUC-PR, Grad-CAM samples

### 5.3 Web Frontend
- **FR-19**: Upload fundus image and receive DR severity prediction with confidence scores
- **FR-20**: Display Grad-CAM heatmap overlay showing regions influencing the prediction
- **FR-21**: Show per-class probability distribution
- **FR-22**: Batch processing capability for screening programs
- **FR-23**: Dashboard with historical prediction analytics
- **FR-24**: Responsive design for desktop and mobile/tablet use in clinical settings

### 5.4 Deployment
- **FR-25**: Model export for serving (TensorFlow Serving, ONNX, or TorchServe)
- **FR-26**: REST API endpoint for prediction
- **FR-27**: Docker containerization for reproducible deployment

---

## 6. Non-Functional Requirements

- **NFR-01**: Training must run on university HPC cluster with NVIDIA H200 GPUs
- **NFR-02**: All code must be version-controlled with phased git pushes
- **NFR-03**: Model checkpoints saved after every epoch (resilience to HPC job interruptions)
- **NFR-04**: Comprehensive logging (TensorBoard, W&B, or equivalent)
- **NFR-05**: Reproducibility: fixed seeds, environment lock files, documented hyperparameters
- **NFR-06**: Ethical: equitable performance across all severity grades; no disproportionate failure on minority classes

---

## 7. Peer Feedback Integration

The following suggestions from course peers have been evaluated and incorporated:

| Peer | Suggestion | Status |
|------|-----------|--------|
| Om Manojkumar Patel | Ben Graham's circle cropping | Incorporated in preprocessing (FR-04) |
| Om Manojkumar Patel | Quadratic Weighted Kappa as primary metric | Adopted as primary evaluation metric |
| Ruthvik Nath Bandari | U-Net / segmentation alongside classification | Exploring via DINOv2 segmentation head (FR-09) |
| Ruthvik Nath Bandari | SMOTE / imbalanced-learn for feature-space oversampling | Included as fallback if augmentation insufficient |
| Ruthvik Nath Bandari | Targeted augmentations centered on optic disc/macula | Incorporated in augmentation strategy (FR-14) |
| Ruthvik Nath Bandari | Confusion matrix validation | Part of evaluation suite (FR-18) |
| Aditya Gulati | Self-supervised pretraining on unlabeled eye images | Addressed via DINOv2 which is self-supervised (FR-09) |
| Aditya Gulati | MixUp augmentation for rare classes | Incorporated (FR-13) |
| Aditya Gulati | Grad-CAM for clinician trust | Incorporated (FR-17, FR-20) |
| Aashay Aamod Gokhale | Prediction threshold tuning for higher sensitivity | Will implement threshold optimization post-training |
| Kapil Varma | Caution with augmentation introducing wrong features | Augmentations calibrated to clinical plausibility (FR-14) |
| Nada Moursi | AUC-PR instead of ROC AUC for imbalanced data | Both metrics reported (FR-18) |
| Sijia Zhan | Quantitative evaluation with sensitivity/AUC metrics | Full evaluation suite implemented (FR-18) |
| Yun-Tang Lin (Cho) | Pretrained models + data augmentation for generalization | Core approach (FR-08, FR-14) |

---

## 8. Out of Scope (v1)

- Real-time video fundus analysis
- Integration with EHR/EMR systems
- FDA/CE regulatory submission
- Multi-language frontend support
- Mobile app (native iOS/Android)
- External dataset cross-validation beyond Messidor-2

---

## 9. References

- Gulshan, V. et al. (2016). JAMA, 316(22), 2402–2410.
- Abràmoff, M. D. et al. (2016). Investigative Ophthalmology & Visual Science, 57(13), 5200–5206.
- Krause, J. et al. (2018). Ophthalmology, 125(8), 1264–1272.
- Tan, M. & Le, Q. V. (2019). ICML.
- He, K. et al. (2016). CVPR, 770–778.
- Graham, B. (2015). Kaggle DR Competition Report. University of Warwick.
- Selvaraju, R. R. et al. (2017). ICCV, 618–626.
- Shorten, C. & Khoshgoftaar, T. M. (2019). Journal of Big Data, 6(1), 1–48.
- Dosovitskiy, A. et al. (2021). ICLR.
- Chetoui, M. & Akhloufi, M. A. (2020). Journal of Medical Imaging, 7(4), 044503.
- Obermeyer, Z. et al. (2019). Science, 366(6464), 447–453.
- Holzinger, A. et al. (2019). WIREs DMKD, 9(4), e1312.
- Esteva, A. et al. (2017). Nature, 542(7639), 115–118.
- Goodfellow, I. et al. (2016). Deep Learning. MIT Press.
- LeCun, Y. et al. (1998). Proceedings of the IEEE, 86(11), 2278–2324.
