"""Entry point: Full evaluation suite.

Usage:
    python scripts/run_evaluation.py --config configs/train_efficientnet.yaml --checkpoint checkpoints/efficientnet_b4/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.augmentation import get_val_transforms
from src.data.dataset import create_dataloaders, create_splits
from src.evaluation.evaluate import compare_models, evaluate_model
from src.evaluation.gradcam import generate_gradcam_samples
from src.evaluation.threshold_opt import optimize_thresholds
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything

import numpy as np
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description="DR Model Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--skip-threshold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("dr_detection", log_dir="logs")
    seed_everything(args.seed)

    config = load_config(args.config)
    data_config = load_config(config.get("data_config", "configs/data_config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.model.name

    logger.info("Evaluating %s from %s", model_name, args.checkpoint)

    # Build model
    if "efficientnet" in model_name:
        from src.models.efficientnet import build_efficientnet
        model = build_efficientnet(config)
    elif "dinov2" in model_name:
        from src.models.dinov2 import build_dinov2
        model = build_dinov2(config)
    elif "resnet" in model_name:
        from src.models.resnet_baseline import build_resnet
        model = build_resnet(config)
    else:
        logger.error("Unknown model: %s", model_name)
        sys.exit(1)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    # Data
    labels_path = Path(data_config.paths.preprocessed_data) / "aptos2019" / "labels.csv"
    if not labels_path.exists():
        labels_path = Path(data_config.datasets.aptos2019.local_path) / "labels.csv"

    _, _, test_df = create_splits(
        labels_path,
        train_ratio=data_config.splits.train,
        val_ratio=data_config.splits.val,
        test_ratio=data_config.splits.test,
        seed=data_config.splits.random_seed,
    )

    image_size = config.model.image_size
    val_transform = get_val_transforms(config.augmentation, image_size)

    data_dir = Path(data_config.paths.preprocessed_data) / "aptos2019"
    if not data_dir.exists():
        data_dir = Path(data_config.datasets.aptos2019.local_path) / "images"

    from src.data.dataset import DRDataset
    from torch.utils.data import DataLoader

    test_dataset = DRDataset(
        image_ids=test_df["image_id"].astype(str).tolist(),
        labels=test_df["label"].tolist(),
        images_dir=str(data_dir),
        transform=val_transform,
        use_cache=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    output_dir = Path(args.output_dir or f"outputs/{model_name}/evaluation")

    # Full evaluation
    metrics = evaluate_model(model, test_loader, device, output_dir, model_name)

    # Threshold optimization
    if not args.skip_threshold:
        # Collect predictions for threshold optimization
        all_labels, all_probs = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                output = model(images.to(device))
                if isinstance(output, dict):
                    output = output["logits"]
                all_probs.append(F.softmax(output, dim=-1).cpu().numpy())
                all_labels.append(labels.numpy())

        y_true = np.concatenate(all_labels)
        y_probs = np.concatenate(all_probs)
        optimize_thresholds(y_true, y_probs, output_dir=output_dir)

    # Grad-CAM
    if not args.skip_gradcam:
        generate_gradcam_samples(model, test_loader, model_name, output_dir, device)

    logger.info("Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
