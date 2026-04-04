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
