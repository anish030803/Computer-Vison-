# CLAUDE.md — Instructions for Claude Code Extension

## Project Context

This is an **Automated Diabetic Retinopathy Detection & Grading System** using deep learning on retinal fundus images. It classifies images into 5 severity grades (No DR → Proliferative DR) using EfficientNet-B4, DINOv2, and ResNet-50 models. The system includes a training pipeline (run on university HPC with H200 GPUs), a FastAPI inference server, and a Next.js web frontend.

**Repository:** https://github.com/anish030803/Computer-Vison-.git

---

## Critical Project Documents — READ THESE FIRST

Before starting ANY work, read the relevant documents:

| Document | Path | When to Read |
|----------|------|-------------|
| Product Requirements | `docs/PRD.md` | Before any feature work |
| Technical Requirements | `docs/TRD.md` | Before any implementation |
| Task Tracker | `tasks/todo.md` | Before starting any task |
| Lessons Learned | `tasks/lessons.md` | At session start |

---

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review `tasks/lessons.md` at session start for relevant project context

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
- **SOLID Principles**: Apply across both Python and TypeScript code.

---

## Project-Specific Rules

### Data Pipeline Rules
- **NEVER skip the cleaning loop.** Data cleaning runs iteratively until ALL quality gates pass. Do not proceed to preprocessing until cleaning is verified.
- **ALWAYS verify data integrity** before and after every transformation step. Log checksums, counts, and distributions.
- **Download data from the best available public sources.** Search for APTOS 2019, EyePACS, Messidor-2, IDRiD, DDR datasets. Verify via checksums.
- **Set hyperparameters AFTER data cleaning.** Class weights, batch sizes, augmentation intensity, and focal loss gamma must be computed from the actual cleaned dataset statistics — never hardcode them.
- **Cache preprocessed data.** Save preprocessed images as .npy or HDF5 to avoid recomputation. Verify cache integrity on load.

### Model & Training Rules
- **Two-phase training is mandatory.** Phase 1: warmup with frozen backbone. Phase 2: fine-tune with partially unfrozen backbone. Never skip the warmup phase.
- **Learning rate warmup before main training.** Linear warmup for the first 5–10% of total epochs in each phase.
- **Overfitting prevention is non-negotiable.** Every training run must include: early stopping, dropout, label smoothing, weight decay, and data augmentation. Check val loss vs train loss divergence every epoch.
- **Checkpoint every epoch.** On HPC, jobs can be killed at any time. Save full training state (model, optimizer, scheduler, epoch, best metric) after every epoch.
- **Mixed precision (BF16) by default.** H200 GPUs support BF16 natively. Always enable AMP.
- **DINOv2 is a first-class model**, not an afterthought. It should have its own config, training script, and evaluation. Consider it for both classification and segmentation.
- **Transfer learning is the core approach.** Use ImageNet pretrained weights for EfficientNet-B4/ResNet-50 and self-supervised pretrained weights for DINOv2. Never train from scratch.

### Evaluation Rules
- **QWK is the primary metric.** Report it prominently. Target ≥ 0.85.
- **Per-class recall for Severe and Proliferative DR must be tracked separately.** These are clinically critical — missing severe cases is the worst failure mode.
- **Always generate Grad-CAM visualizations.** They're required for clinical trust and for verifying the model attends to actual pathology, not artifacts.
- **Report AUC-PR alongside AUC-ROC** for minority classes (Severe, Proliferative DR).

### Frontend Rules
- **Next.js with App Router** — no Pages Router
- **Bun** for package management and compilation
- **TypeScript strict mode** — no `any` types
- **Tailwind CSS** for all styling — no CSS modules, no styled-components
- **React Hook Form + Zod** for all forms and validation
- **React Query (TanStack Query)** for all server state
- **Animate UI + Motion** for animations and transitions
- **Sharp** for image optimization in Next.js
- **oxlint** for linting — run before every commit
- **SOLID principles** for component architecture:
  - Single Responsibility: one component = one job
  - Open/Closed: extend via props/composition, don't modify
  - Liskov Substitution: components should be replaceable
  - Interface Segregation: small, focused prop interfaces
  - Dependency Inversion: depend on abstractions (hooks, contexts), not implementations

### Git Rules
- **Every phase = one git push** to the repository
- **Tag each phase** with the version specified in `tasks/todo.md`
- **Never push broken code.** Run verification checks before every push.
- **Meaningful commit messages**: `phase-N: description of what was done`

---

## Compute Environment

- **Training**: University HPC with NVIDIA H200 GPUs (80GB HBM3)
- **Job Scheduler**: SLURM
- **Python**: 3.11+
- **Deep Learning**: PyTorch 2.x (primary), with timm for model zoo
- **Data can be large**: H200 has ample memory — use generous batch sizes, full-resolution images
- **Mixed Precision**: BF16 enabled by default

---

## Key File Locations

```
configs/                  → All YAML configs (data, training, HPC)
src/data/                 → Data pipeline (download, clean, preprocess, augment)
src/models/               → Model architectures (EfficientNet, DINOv2, ResNet, ensemble)
src/training/             → Training loop, losses, schedulers, callbacks, metrics
src/evaluation/           → Evaluation suite, Grad-CAM, threshold optimization
scripts/                  → Entry point scripts (run_cleaning, run_training, etc.)
server/                   → FastAPI inference server
frontend/                 → Next.js web application
tasks/todo.md             → Current task tracker (update as you go)
tasks/lessons.md          → Lessons learned (update after every mistake)
```

---

## Common Gotchas

1. **Kaggle API needs credentials.** Ensure `~/.kaggle/kaggle.json` exists on HPC before running data download.
2. **APTOS 2019 class distribution is heavily imbalanced.** ~49% No DR, <3% Severe DR. Class weights are essential.
3. **Ben Graham's preprocessing requires Gaussian blur kernel size proportional to image width.** Typically sigma = 10% of image width (e.g., sigma=30 for 300px).
4. **DINOv2 expects 518×518 input by default** (ViT-L/14 with patch size 14). Don't resize to 380×380 for DINOv2.
5. **EfficientNet-B4 native resolution is 380×380.** Using a different resolution will degrade performance.
6. **Grad-CAM requires the model to have spatial feature maps.** For DINOv2 (ViT), use attention rollout or attention-based visualization instead.
7. **SLURM jobs have time limits.** Always save checkpoints frequently and support resume from checkpoint.
8. **The APTOS 2019 test set has no labels.** Use the training set with stratified splits for all evaluation. Messidor-2 is the external validation set.
