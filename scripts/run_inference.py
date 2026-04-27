"""Entry point: Single image inference.

Usage:
    python scripts/run_inference.py --image path/to/fundus.jpg --checkpoint checkpoints/efficientnet_b4/best.pt
    python scripts/run_inference.py --image path/to/fundus.jpg --config configs/train_efficientnet.yaml --checkpoint best.pt --gradcam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocessing import ben_graham_preprocess
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def main() -> None:
    parser = argparse.ArgumentParser(description="DR Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to fundus image")
    parser.add_argument("--config", type=str, default="configs/train_efficientnet.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlay")
    parser.add_argument("--output", type=str, default=None, help="Save result image to this path")
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model_name = config.model.name
    image_size = config.model.image_size

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
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        sys.exit(1)

    # Ben Graham preprocessing
    preprocessed = ben_graham_preprocess(img, target_size=image_size)

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (preprocessed - mean) / std

    # To tensor
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict):
            output = output["logits"]
        probs = F.softmax(output, dim=-1)[0].cpu().numpy()

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    result = {
        "image": args.image,
        "prediction": {
            "class": pred_class,
            "label": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        },
        "model": model_name,
    }

    print(json.dumps(result, indent=2))

    # Grad-CAM
    if args.gradcam and args.output:
        from src.evaluation.gradcam import GradCAM, get_target_layer, overlay_heatmap

        if "dinov2" not in model_name:
            target_layer = get_target_layer(model, model_name)
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(tensor, target_class=pred_class)
            gradcam.cleanup()

            # Overlay on original image
            img_resized = cv2.resize(img, (image_size, image_size))
            overlay = overlay_heatmap(img_resized, heatmap)
            cv2.imwrite(args.output, overlay)
            print(f"Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
