# Lessons Learned

> Update this file after EVERY correction from the user. Write rules that prevent the same mistake from happening again. Review at session start.

---

## Format

```
### [Date] — [Category]: [Brief description]
**Mistake**: What went wrong
**Root cause**: Why it happened
**Rule**: The rule to prevent recurrence
**Applies to**: Which files/modules this affects
```

---

### 2026-04-03 — Data Pipeline: Quality thresholds too aggressive for fundus images
**Mistake**: Cleaning removed 99.7% of images (2,346/2,354). Only 8 survived.
**Root cause**: Laplacian sharpness threshold (100.0), brightness floor (30), and contrast floor (20) were set for generic natural images. Fundus images have dark circular backgrounds, soft focus, and low overall brightness/contrast — these are features, not defects.
**Rule**: Always calibrate quality thresholds by first running a quality stats scan on a sample of the actual dataset before setting thresholds. For fundus images: sharpness ≥ 5, brightness ≥ 5, contrast ≥ 5.
**Applies to**: `configs/data_config.yaml`, `src/data/cleaning.py`

### 2026-04-03 — Data Pipeline: Duplicate detection hamming threshold too loose
**Mistake**: Hamming threshold of 5 flagged 414 images (14%) as duplicates — likely too many false positives.
**Root cause**: Fundus images of the same eye can look very similar but are different patients/visits. pHash with hamming < 5 is too loose for medical images with similar structure.
**Rule**: Use hamming_threshold ≤ 3 for medical imaging datasets. Consider adding image metadata (patient ID) to dedup logic when available.
**Applies to**: `configs/data_config.yaml`

### 2026-04-05 — Training: Phase 2 checkpointer reset overwrites better Phase 1 model
**Mistake**: `checkpointer.reset()` between Phase 1→2 set `best_value = None`, so Phase 2 epoch 1 always became the new "best" — even when Phase 1's best QWK (0.7387) was higher than Phase 2's best (0.7048). The final `best.pt` was the inferior model.
**Root cause**: Both `early_stopping.reset()` and `checkpointer.reset()` were called at phase transition. Early stopping needs a reset (fresh patience), but the checkpointer should retain Phase 1's best so Phase 2 only overwrites if it truly improves.
**Rule**: Never reset checkpointer best_value between training phases. Only reset early stopping. The best model across ALL phases should be preserved.
**Applies to**: `src/training/trainer.py`, `src/training/callbacks.py`

### 2026-04-05 — HPC: Disk quota prevents git pull, causes stale code execution
**Mistake**: `git pull` on HPC failed silently (`cannot open .git/FETCH_HEAD: Disk quota exceeded`). Training ran with stale code — epoch checkpoints and TensorBoard were still being saved despite the "fix" commit.
**Root cause**: Checkpoints, logs, and TensorBoard event files accumulated and exhausted disk quota. The subsequent `git pull` failure was easy to miss in the log output.
**Rule**: Always check `git pull` exit status before running training. Clean up outputs/logs/checkpoints BEFORE pulling. Consider adding a pre-training disk quota check.
**Applies to**: HPC workflow, all training scripts

### 2026-04-05 — Config: num_workers=16 exceeds HPC node limit of 8
**Mistake**: DataLoader warned that 16 workers exceeds the system's suggested max of 8, which can cause slowness or freezes.
**Root cause**: num_workers was set for a different hardware config. HPC nodes have 8 available.
**Rule**: Set num_workers to match the HPC node's CPU core count (8 on Explorer cluster). Check `nproc` on the target machine.
**Applies to**: `configs/train_efficientnet.yaml`, `configs/train_dinov2.yaml`, `configs/train_resnet.yaml`
