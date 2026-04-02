"""Entry point: Full training pipeline.

Usage:
    python scripts/run_training.py --config configs/train_efficientnet.yaml
    python scripts/run_training.py --config configs/train_dinov2.yaml --resume checkpoints/dinov2_vit_l/latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import create_dataloaders, create_splits
from src.training.trainer import Trainer
from src.utils.config import load_config, load_and_merge_configs
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DR Model Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override preprocessed data directory",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    data_config_path = config.get("data_config", "configs/data_config.yaml")
    data_config = load_config(data_config_path)

    # Setup
    seed = args.seed or config.get("seed", 42)
    logger = setup_logger("dr_detection", log_dir="logs")
    seed_everything(seed)

    logger.info("=" * 60)
    logger.info("DR Model Training")
    logger.info("Config: %s", args.config)
    logger.info("Model: %s", config.model.name)
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Setup ---
    # Find dataset directory
    data_dir = args.data_dir
    if data_dir is None:
        # Use preprocessed data if available, otherwise raw
        preprocessed = Path(data_config.paths.preprocessed_data) / "aptos2019"
        raw = Path(data_config.datasets.aptos2019.local_path)
        if preprocessed.exists():
            data_dir = str(preprocessed)
            use_cache = True
        elif raw.exists():
            data_dir = str(raw / "images")
            use_cache = False
        else:
            logger.error(
                "No data found. Run scripts/run_cleaning.py first.\n"
                "Expected: %s or %s", preprocessed, raw,
            )
            sys.exit(1)
    else:
        use_cache = Path(data_dir).glob("*.npy").__next__() is not None

    logger.info("Data directory: %s (cache=%s)", data_dir, use_cache)

    # Labels
    labels_path = Path(data_dir) / "labels.csv"
    if not labels_path.exists():
        # Try parent directory
        labels_path = Path(data_dir).parent / "labels.csv"
    if not labels_path.exists():
        logger.error("Labels not found at %s", labels_path)
        sys.exit(1)

    # Splits
    split_cfg = data_config.splits
    train_df, val_df, test_df = create_splits(
        labels_path,
        train_ratio=split_cfg.train,
        val_ratio=split_cfg.val,
        test_ratio=split_cfg.test,
        seed=split_cfg.random_seed,
    )

    # Augmentation transforms
    image_size = config.model.image_size
    aug_config = config.augmentation
    train_transform = get_train_transforms(aug_config, image_size)
    val_transform = get_val_transforms(aug_config, image_size)

    # Load dynamic class weights if available
    class_weights = None
    hp_path = Path(data_config.paths.reports_dir) / "dynamic_hyperparameters.json"
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
        if "class_weights" in hp:
            class_weights = [hp["class_weights"][str(i)] for i in range(5)]
            logger.info("Loaded dynamic class weights: %s", class_weights)

    # DataLoaders
    batch_size = config.training.phase1.batch_size
    hw_cfg = config.get("hardware", None)
    num_workers = hw_cfg.get("num_workers", 4) if hw_cfg else 4

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        images_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        use_cache=use_cache,
        class_weights={i: w for i, w in enumerate(class_weights)} if class_weights else None,
    )

    # --- Model Setup ---
    model_name = config.model.name
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

    # --- Train ---
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        resume_checkpoint=args.resume,
    )

    history = trainer.train()

    # --- Save History ---
    output_dir = Path(config.get("output_dir", f"outputs/{model_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Training history saved to %s", history_path)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
