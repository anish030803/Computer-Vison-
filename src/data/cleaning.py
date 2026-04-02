"""Iterative data cleaning loop with quality gates.

Runs 5 passes until all gates pass:
1. File integrity (corrupt/truncated images)
2. Duplicate detection (perceptual hashing)
3. Quality assessment (sharpness, brightness, contrast)
4. Resolution & format normalization
5. Label verification
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class CleaningReport:
    """Accumulates results from each cleaning pass."""

    def __init__(self) -> None:
        self.iterations: list[dict[str, Any]] = []
        self.removed_files: list[dict[str, str]] = []
        self.total_removed = 0
        self.start_time = datetime.now().isoformat()

    def add_iteration(self, iteration: int, pass_results: dict[str, Any]) -> None:
        self.iterations.append({"iteration": iteration, **pass_results})

    def add_removed(self, filepath: str, reason: str, pass_name: str) -> None:
        self.removed_files.append({
            "file": filepath,
            "reason": reason,
            "pass": pass_name,
        })
        self.total_removed += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "total_iterations": len(self.iterations),
            "total_removed": self.total_removed,
            "iterations": self.iterations,
            "removed_files": self.removed_files,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Cleaning report saved to %s", path)


def run_cleaning_loop(
    dataset_path: str | Path,
    config: Config,
    max_iterations: int = 5,
) -> CleaningReport:
    """Run the iterative cleaning loop until all quality gates pass.

    Args:
        dataset_path: Path to dataset (must contain images/ and labels.csv).
        config: Data config with cleaning thresholds.
        max_iterations: Safety limit on iterations.

    Returns:
        CleaningReport with full audit trail.
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_path = dataset_path / "labels.csv"
    cleaning_cfg = config.cleaning

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    report = CleaningReport()

    for iteration in range(1, max_iterations + 1):
        logger.info("=== Cleaning iteration %d ===", iteration)
        pass_results: dict[str, Any] = {}
        all_gates_passed = True

        # Pass 1: File integrity
        p1_result = _pass_file_integrity(images_dir, report)
        pass_results["file_integrity"] = p1_result
        if not p1_result["gate_passed"]:
            all_gates_passed = False

        # Pass 2: Duplicate detection
        if cleaning_cfg.duplicate_detection.enabled:
            p2_result = _pass_duplicate_detection(
                images_dir,
                cleaning_cfg.duplicate_detection.hamming_threshold,
                report,
            )
            pass_results["duplicate_detection"] = p2_result
            if not p2_result["gate_passed"]:
                all_gates_passed = False

        # Pass 3: Quality assessment
        if cleaning_cfg.quality_assessment.enabled:
            p3_result = _pass_quality_assessment(
                images_dir,
                cleaning_cfg.quality_assessment,
                report,
            )
            pass_results["quality_assessment"] = p3_result
            if not p3_result["gate_passed"]:
                all_gates_passed = False

        # Pass 4: Resolution & format
        p4_result = _pass_resolution_format(
            images_dir,
            cleaning_cfg.resolution,
            report,
        )
        pass_results["resolution_format"] = p4_result
        if not p4_result["gate_passed"]:
            all_gates_passed = False

        # Pass 5: Label verification
        if cleaning_cfg.label_verification.enabled and labels_path.exists():
            p5_result = _pass_label_verification(images_dir, labels_path, report)
            pass_results["label_verification"] = p5_result
            if not p5_result["gate_passed"]:
                all_gates_passed = False

        # Count remaining images
        remaining = len(list(images_dir.glob("*")))
        pass_results["images_remaining"] = remaining
        report.add_iteration(iteration, pass_results)

        logger.info(
            "Iteration %d complete: %d images remaining, all gates passed: %s",
            iteration, remaining, all_gates_passed,
        )

        if all_gates_passed:
            logger.info("All quality gates passed after %d iteration(s)", iteration)
            break
    else:
        logger.warning(
            "Reached max iterations (%d) without all gates passing", max_iterations
        )

    # Save report
    report_path = dataset_path / "cleaning_report.json"
    report.save(report_path)

    return report


def _pass_file_integrity(images_dir: Path, report: CleaningReport) -> dict:
    """Pass 1: Remove corrupt or truncated images."""
    corrupt_count = 0
    checked = 0

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        checked += 1

        try:
            img = Image.open(img_path)
            img.verify()
            # Re-open after verify (verify closes the file)
            img = Image.open(img_path)
            img.load()
        except Exception as e:
            logger.debug("Corrupt image: %s (%s)", img_path.name, e)
            report.add_removed(str(img_path), f"corrupt: {e}", "file_integrity")
            img_path.unlink()
            corrupt_count += 1

    logger.info("File integrity: checked=%d, corrupt=%d", checked, corrupt_count)
    return {
        "checked": checked,
        "corrupt_removed": corrupt_count,
        "gate_passed": corrupt_count == 0,
    }


def _pass_duplicate_detection(
    images_dir: Path,
    hamming_threshold: int,
    report: CleaningReport,
) -> dict:
    """Pass 2: Remove near-duplicate images using perceptual hashing."""
    hashes: dict[str, tuple[imagehash.ImageHash, Path]] = {}
    duplicates_removed = 0

    image_files = sorted(images_dir.iterdir())

    for img_path in image_files:
        if not img_path.is_file():
            continue

        try:
            img = Image.open(img_path)
            phash = imagehash.phash(img)
        except Exception:
            continue

        # Check against existing hashes
        is_duplicate = False
        for existing_key, (existing_hash, existing_path) in hashes.items():
            distance = phash - existing_hash
            if distance < hamming_threshold:
                # Keep the larger (presumably higher quality) file
                if img_path.stat().st_size >= existing_path.stat().st_size:
                    # Remove existing, keep current
                    report.add_removed(
                        str(existing_path),
                        f"duplicate of {img_path.name} (hamming={distance})",
                        "duplicate_detection",
                    )
                    existing_path.unlink()
                    hashes[existing_key] = (phash, img_path)
                else:
                    # Remove current, keep existing
                    report.add_removed(
                        str(img_path),
                        f"duplicate of {existing_path.name} (hamming={distance})",
                        "duplicate_detection",
                    )
                    img_path.unlink()
                duplicates_removed += 1
                is_duplicate = True
                break

        if not is_duplicate:
            hashes[img_path.name] = (phash, img_path)

    logger.info("Duplicate detection: removed=%d", duplicates_removed)
    return {
        "duplicates_removed": duplicates_removed,
        "gate_passed": duplicates_removed == 0,
    }


def _pass_quality_assessment(
    images_dir: Path,
    quality_cfg: Config,
    report: CleaningReport,
) -> dict:
    """Pass 3: Remove images below quality thresholds."""
    removed = 0
    checked = 0
    quality_scores: list[float] = []

    min_sharpness = quality_cfg.min_sharpness
    min_brightness = quality_cfg.min_brightness
    max_brightness = quality_cfg.max_brightness
    min_contrast = quality_cfg.min_contrast

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        checked += 1

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Sharpness via Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_scores.append(sharpness)

            # Brightness (mean pixel value)
            brightness = gray.mean()

            # Contrast (std deviation)
            contrast = gray.std()

            reasons = []
            if sharpness < min_sharpness:
                reasons.append(f"sharpness={sharpness:.1f}<{min_sharpness}")
            if brightness < min_brightness:
                reasons.append(f"brightness={brightness:.1f}<{min_brightness}")
            if brightness > max_brightness:
                reasons.append(f"brightness={brightness:.1f}>{max_brightness}")
            if contrast < min_contrast:
                reasons.append(f"contrast={contrast:.1f}<{min_contrast}")

            if reasons:
                report.add_removed(str(img_path), "; ".join(reasons), "quality_assessment")
                img_path.unlink()
                removed += 1

        except Exception as e:
            logger.debug("Quality check failed for %s: %s", img_path.name, e)

    logger.info("Quality assessment: checked=%d, removed=%d", checked, removed)
    return {
        "checked": checked,
        "removed": removed,
        "mean_sharpness": float(np.mean(quality_scores)) if quality_scores else 0,
        "gate_passed": removed == 0,
    }


def _pass_resolution_format(
    images_dir: Path,
    resolution_cfg: Config,
    report: CleaningReport,
) -> dict:
    """Pass 4: Remove images below minimum resolution, normalize formats."""
    removed = 0
    checked = 0
    min_w = resolution_cfg.min_width
    min_h = resolution_cfg.min_height
    allowed = set(resolution_cfg.allowed_formats)

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        checked += 1

        ext = img_path.suffix.lower().lstrip(".")

        # Check format
        if ext not in allowed:
            report.add_removed(str(img_path), f"unsupported format: {ext}", "resolution_format")
            img_path.unlink()
            removed += 1
            continue

        # Check resolution
        try:
            img = Image.open(img_path)
            w, h = img.size

            if w < min_w or h < min_h:
                report.add_removed(
                    str(img_path),
                    f"resolution {w}x{h} below minimum {min_w}x{min_h}",
                    "resolution_format",
                )
                img_path.unlink()
                removed += 1
        except Exception as e:
            logger.debug("Resolution check failed for %s: %s", img_path.name, e)

    logger.info("Resolution/format: checked=%d, removed=%d", checked, removed)
    return {
        "checked": checked,
        "removed": removed,
        "gate_passed": removed == 0,
    }


def _pass_label_verification(
    images_dir: Path,
    labels_path: Path,
    report: CleaningReport,
) -> dict:
    """Pass 5: Verify all images have labels and vice versa."""
    df = pd.read_csv(labels_path)
    labeled_ids = set(df["image_id"].astype(str))

    # Get image IDs (filename without extension)
    image_ids = {f.stem for f in images_dir.iterdir() if f.is_file()}

    # Images without labels
    unlabeled = image_ids - labeled_ids
    # Labels without images (orphan labels)
    orphan_labels = labeled_ids - image_ids

    removed = 0
    for img_id in unlabeled:
        # Find and remove the image file
        for img_path in images_dir.glob(f"{img_id}.*"):
            report.add_removed(str(img_path), "no label found", "label_verification")
            img_path.unlink()
            removed += 1

    # Clean orphan labels from CSV
    if orphan_labels:
        df = df[~df["image_id"].astype(str).isin(orphan_labels)]
        df.to_csv(labels_path, index=False)
        logger.info("Removed %d orphan labels from CSV", len(orphan_labels))

    # Log class distribution
    if not df.empty and "label" in df.columns:
        dist = df["label"].value_counts().sort_index().to_dict()
        logger.info("Class distribution after label verification: %s", dist)

    logger.info("Label verification: unlabeled=%d, orphan=%d", len(unlabeled), len(orphan_labels))
    return {
        "unlabeled_images_removed": len(unlabeled),
        "orphan_labels_removed": len(orphan_labels),
        "gate_passed": len(unlabeled) == 0 and len(orphan_labels) == 0,
    }
