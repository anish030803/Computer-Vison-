"""Upload trained models to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py --repo-id your-username/dr-detection --model efficientnet_b4
    python scripts/upload_to_huggingface.py --repo-id your-username/dr-detection --model resnet50
    python scripts/upload_to_huggingface.py --repo-id your-username/dr-detection --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def upload_model(repo_id: str, model_name: str, token: str | None = None) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    checkpoint_path = Path(f"checkpoints/{model_name}/best.pt")
    config_path = Path(f"configs/train_{model_name.replace('_b4', '')}.yaml")

    if not checkpoint_path.exists():
        # Try alternate config naming
        for cfg in Path("configs").glob(f"train_*{model_name.split('_')[0]}*.yaml"):
            config_path = cfg
            break

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Create repo if needed
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Note: {e}")

    print(f"Uploading {model_name} to {repo_id}...")

    # Upload checkpoint
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=f"{model_name}/best.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Uploaded: {model_name}/best.pt")

    # Upload config
    if config_path.exists():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo=f"{model_name}/config.yaml",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded: {model_name}/config.yaml")

    # Upload training history if available
    history_path = Path(f"outputs/{model_name}/training_history.json")
    if history_path.exists():
        api.upload_file(
            path_or_fileobj=str(history_path),
            path_in_repo=f"{model_name}/training_history.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded: {model_name}/training_history.json")

    # Upload CV results if available
    for cv_dir in Path("outputs").glob(f"{model_name}/cross_validation_*"):
        cv_results = cv_dir / "cv_results.json"
        if cv_results.exists():
            api.upload_file(
                path_or_fileobj=str(cv_results),
                path_in_repo=f"{model_name}/cv_results.json",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded: {model_name}/cv_results.json")

    # Create model card
    model_card = f"""---
tags:
  - diabetic-retinopathy
  - medical-imaging
  - classification
  - pytorch
license: mit
---

# DR Detection — {model_name}

Automated Diabetic Retinopathy Detection & Grading model trained on APTOS 2019 dataset.

## Model

- **Architecture**: {model_name}
- **Input size**: 380x380 (EfficientNet/ResNet) or 518x518 (DINOv2)
- **Classes**: 5 (No DR, Mild NPDR, Moderate NPDR, Severe NPDR, Proliferative DR)
- **Preprocessing**: Ben Graham's method

## Training

- Two-phase training: frozen backbone warmup → partial fine-tuning
- Class-weighted cross-entropy with label smoothing
- BF16 mixed precision on NVIDIA A100 80GB

## Usage

```python
import torch
from src.models.efficientnet import build_efficientnet
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint

config = load_config("configs/train_efficientnet.yaml")
model = build_efficientnet(config)
load_checkpoint("checkpoints/efficientnet_b4/best.pt", model)
model.eval()
```
"""
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Uploaded: README.md")
    print(f"Done! Model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="e.g. your-username/dr-detection")
    parser.add_argument("--model", type=str, help="Model name (efficientnet_b4, resnet50)")
    parser.add_argument("--all", action="store_true", help="Upload all available models")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    if args.all:
        for model_dir in Path("checkpoints").iterdir():
            if (model_dir / "best.pt").exists():
                upload_model(args.repo_id, model_dir.name, args.token)
    elif args.model:
        upload_model(args.repo_id, args.model, args.token)
    else:
        print("Specify --model or --all")


if __name__ == "__main__":
    main()
