"""Entry point: Data download, cleaning loop, preprocessing, and validation.

Usage:
    python scripts/run_cleaning.py --config configs/data_config.yaml
    python scripts/run_cleaning.py --config configs/data_config.yaml --datasets aptos2019
    python scripts/run_cleaning.py --config configs/data_config.yaml --skip-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaning import run_cleaning_loop
from src.data.download import download_all_datasets
from src.data.preprocessing import preprocess_dataset, verify_preprocessed_cache
from src.data.validation import compute_dynamic_hyperparameters, validate_cleaned_dataset
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DR Data Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/data_config.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Specific datasets to process (default: all)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (assume data exists)",
    )
    parser.add_argument(
        "--skip-cleaning", action="store_true",
        help="Skip cleaning step (assume data is clean)",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip preprocessing step",
    )
    parser.add_argument(
        "--target-size", type=int, default=None,
        help="Override target image size",
    )
    parser.add_argument(
        "--method", type=str, default="ben_graham",
        choices=["ben_graham", "clahe", "green_channel"],
        help="Preprocessing method",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    logger = setup_logger("dr_detection", log_dir="logs")
    seed_everything(args.seed)
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("DR Data Pipeline Starting")
    logger.info("Config: %s", args.config)
    logger.info("=" * 60)

    # --- Step 1: Download ---
    if not args.skip_download:
        logger.info("--- Step 1: Download Datasets ---")
        results = download_all_datasets(config, datasets=args.datasets)
        for name, success in results.items():
            status = "OK" if success else "FAILED"
            logger.info("  %s: %s", name, status)
    else:
        logger.info("--- Step 1: Download SKIPPED ---")

    # --- Step 2: Cleaning ---
    datasets_to_process = args.datasets or list(config.datasets.to_dict().keys())

    for ds_name in datasets_to_process:
        ds_cfg = config.datasets[ds_name]
        ds_path = Path(ds_cfg["local_path"]) if isinstance(ds_cfg, dict) else Path(ds_cfg.local_path)

        if not (ds_path / "images").exists():
            logger.warning("Skipping %s — images directory not found at %s", ds_name, ds_path)
            continue

        if not args.skip_cleaning:
            logger.info("--- Step 2: Cleaning %s ---", ds_name)
            report = run_cleaning_loop(ds_path, config)
            logger.info(
                "Cleaning complete: %d iterations, %d files removed",
                len(report.iterations), report.total_removed,
            )

        # --- Step 3: Validation ---
        logger.info("--- Step 3: Validating %s ---", ds_name)
        validation_results = validate_cleaned_dataset(ds_path, config)

        # --- Step 4: Dynamic Hyperparameters ---
        logger.info("--- Step 4: Computing Dynamic Hyperparameters ---")
        hyperparams = compute_dynamic_hyperparameters(ds_path, config)

        # --- Step 5: Preprocessing ---
        if not args.skip_preprocessing:
            target_size = args.target_size or config.image_sizes.efficientnet_b4
            output_path = Path(config.paths.preprocessed_data) / ds_name

            logger.info(
                "--- Step 5: Preprocessing %s (method=%s, size=%d) ---",
                ds_name, args.method, target_size,
            )
            preprocess_dataset(ds_path, output_path, config, target_size, args.method)

            # Verify cache
            expected_shape = (target_size, target_size, 3)
            cache_result = verify_preprocessed_cache(output_path, expected_shape)
            if cache_result["all_valid"]:
                logger.info("Cache verification PASSED for %s", ds_name)
            else:
                logger.warning("Cache verification FAILED for %s: %s", ds_name, cache_result)

    logger.info("=" * 60)
    logger.info("DR Data Pipeline Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
