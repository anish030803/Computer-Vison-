"""Entry point: K-fold cross-validation training.

Usage:
    python scripts/run_cross_validation.py --config configs/train_efficientnet.yaml --folds 5
    python scripts/run_cross_validation.py --config configs/train_resnet.yaml --folds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.training.cross_validation import CrossValidationTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DR K-Fold Cross-Validation")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    data_config_path = config.get("data_config", "configs/data_config.yaml")
    data_config = load_config(data_config_path)

    seed = args.seed or config.get("seed", 42)
    logger = setup_logger("dr_detection", log_dir="logs")
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.model.name

    logger.info("=" * 60)
    logger.info("K-Fold Cross-Validation")
    logger.info("Model: %s | Folds: %d | Device: %s", model_name, args.folds, device)
    logger.info("=" * 60)

    # Resolve data paths
    data_dir = args.data_dir
    use_cache = True
    if data_dir is None:
        preprocessed = Path(data_config.paths.preprocessed_data) / "aptos2019"
        raw = Path(data_config.datasets.aptos2019.local_path)
        if preprocessed.exists():
            data_dir = str(preprocessed)
        elif raw.exists():
            data_dir = str(raw / "images")
            use_cache = False
        else:
            logger.error("No data found. Run scripts/run_cleaning.py first.")
            sys.exit(1)

    # Labels
    labels_path = Path(data_dir) / "labels.csv"
    if not labels_path.exists():
        labels_path = Path(data_dir).parent / "labels.csv"
    if not labels_path.exists():
        logger.error("Labels not found at %s", labels_path)
        sys.exit(1)

    # Load dynamic class weights if available
    class_weights = None
    hp_path = Path(data_config.paths.reports_dir) / "dynamic_hyperparameters.json"
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
        if "class_weights" in hp:
            class_weights = [hp["class_weights"][str(i)] for i in range(5)]
            logger.info("Loaded dynamic class weights: %s", class_weights)

    # Output dir
    output_dir = args.output_dir or f"outputs/{model_name}/cross_validation_{args.folds}fold"

    # Run cross-validation
    cv_trainer = CrossValidationTrainer(
        config=config,
        labels_path=labels_path,
        images_dir=data_dir,
        n_folds=args.folds,
        device=device,
        class_weights=class_weights,
        use_cache=use_cache,
        output_dir=output_dir,
    )

    results = cv_trainer.run()

    # Print summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    agg = results["aggregated"]
    for metric in ["qwk", "accuracy", "macro_f1", "recall_severe_npdr", "recall_proliferative_dr"]:
        if metric in agg:
            s = agg[metric]
            logger.info("  %s: %.4f +/- %.4f", metric, s["mean"], s["std"])

    if results.get("test_metrics"):
        tm = results["test_metrics"]
        logger.info("  --- Test Set (best fold) ---")
        logger.info("  test_qwk: %.4f", tm.get("qwk", 0))
        logger.info("  test_accuracy: %.4f", tm.get("accuracy", 0))

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
