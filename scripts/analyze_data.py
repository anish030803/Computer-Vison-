"""Analyze dataset before and after cleaning.

Usage:
    python scripts/analyze_data.py --data-dir data/raw/aptos2019 --stage raw
    python scripts/analyze_data.py --data-dir data/raw/aptos2019 --stage cleaned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd


def analyze(data_dir: str, stage: str) -> None:
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_path = data_path / "labels.csv"

    print(f"\n{'='*60}")
    print(f"  Data Analysis — {stage.upper()}")
    print(f"{'='*60}\n")

    # --- Labels ---
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        print(f"Total labeled samples: {len(df)}")
        print(f"\nClass Distribution:")
        names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
        dist = df["label"].value_counts().sort_index()
        for label, count in dist.items():
            pct = count / len(df) * 100
            name = names[label] if label < len(names) else f"Class {label}"
            bar = "█" * int(pct / 2)
            print(f"  {label} ({name:20s}): {count:5d} ({pct:5.1f}%) {bar}")

        # Imbalance ratio
        print(f"\n  Imbalance ratio (max/min): {dist.max()/dist.min():.1f}x")
    else:
        print("No labels.csv found")
        df = None

    # --- Images ---
    if images_dir.exists():
        image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
        print(f"\nTotal image files: {len(image_files)}")

        # Check label-image match
        if df is not None:
            image_ids = {f.stem for f in image_files}
            labeled_ids = set(df["image_id"].astype(str))
            matched = image_ids & labeled_ids
            unlabeled = image_ids - labeled_ids
            orphan = labeled_ids - image_ids
            print(f"  Matched (image+label): {len(matched)}")
            print(f"  Images without labels: {len(unlabeled)}")
            print(f"  Labels without images: {len(orphan)}")

        # Sample quality stats (first 200 images)
        sample = image_files[:200]
        widths, heights, sharpness, brightness, contrast = [], [], [], [], []
        formats = {}
        corrupt = 0

        for img_path in sample:
            ext = img_path.suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt += 1
                    continue
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                brightness.append(float(gray.mean()))
                contrast.append(float(gray.std()))
            except Exception:
                corrupt += 1

        print(f"\nImage Quality (sampled {len(sample)} images):")
        print(f"  Corrupt/unreadable: {corrupt}")
        print(f"  Formats: {formats}")
        if widths:
            print(f"  Resolution: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
            print(f"  Avg resolution: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            print(f"  Sharpness  — min: {min(sharpness):.1f}, mean: {np.mean(sharpness):.1f}, max: {max(sharpness):.1f}")
            print(f"  Brightness — min: {min(brightness):.1f}, mean: {np.mean(brightness):.1f}, max: {max(brightness):.1f}")
            print(f"  Contrast   — min: {min(contrast):.1f}, mean: {np.mean(contrast):.1f}, max: {max(contrast):.1f}")

            # Recommend thresholds
            if stage == "raw":
                p5_sharp = np.percentile(sharpness, 2)
                p5_bright = np.percentile(brightness, 2)
                p95_bright = np.percentile(brightness, 98)
                p5_contrast = np.percentile(contrast, 2)
                print(f"\n  Recommended cleaning thresholds (2nd percentile):")
                print(f"    min_sharpness: {p5_sharp:.1f}")
                print(f"    min_brightness: {p5_bright:.1f}")
                print(f"    max_brightness: {p95_bright:.1f}")
                print(f"    min_contrast: {p5_contrast:.1f}")
    else:
        print("No images/ directory found")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="raw", choices=["raw", "cleaned"])
    args = parser.parse_args()
    analyze(args.data_dir, args.stage)
