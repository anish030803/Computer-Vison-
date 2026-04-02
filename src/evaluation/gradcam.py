"""Grad-CAM visualizations for model explainability.

Generates heatmap overlays showing regions influencing predictions.
For CNNs (EfficientNet, ResNet): standard Grad-CAM on last conv layer.
For ViTs (DINOv2): attention map visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("dr_detection")

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNNs.

    Targets the last convolutional layer to generate spatial attention maps.
    Works with EfficientNet-B4 and ResNet-50.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image.

        Args:
            image: Input tensor (1, C, H, W).
            target_class: Class to explain. If None, uses predicted class.

        Returns:
            Heatmap as numpy array (H, W) in [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(image)
        if isinstance(output, dict):
            output = output["logits"]

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def cleanup(self) -> None:
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Get the appropriate target layer for Grad-CAM.

    Args:
        model: The model.
        model_name: Model identifier.

    Returns:
        Target layer module.
    """
    if "efficientnet" in model_name:
        # Last block of EfficientNet backbone
        return model.backbone.conv_head
    elif "resnet" in model_name:
        # Last layer of ResNet backbone
        return model.backbone.layer4[-1]
    else:
        raise ValueError(f"No Grad-CAM target layer defined for {model_name}")


def generate_attention_map(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> np.ndarray:
    """Generate attention map for Vision Transformers (DINOv2).

    Uses attention weights from the last transformer block.

    Args:
        model: DINOv2 model.
        image: Input tensor (1, C, H, W).

    Returns:
        Attention map as numpy array (H, W) in [0, 1].
    """
    model.eval()

    # Get attention from last block
    with torch.no_grad():
        # Forward through backbone to get features
        output = model.backbone.forward_features(image)

        if isinstance(output, dict):
            patch_tokens = output.get("x_norm_patchtokens")
            if patch_tokens is not None:
                # Use norm of patch tokens as attention proxy
                attention = patch_tokens.norm(dim=-1)  # (1, num_patches)
            else:
                attention = output["x_norm_clstoken"].unsqueeze(1)
        else:
            # Fallback: use patch token norms
            patch_tokens = output[:, 1:]  # Remove CLS token
            attention = patch_tokens.norm(dim=-1)

    # Reshape to spatial grid
    num_patches = attention.shape[1]
    grid_size = int(num_patches ** 0.5)
    attention = attention.reshape(1, 1, grid_size, grid_size)

    # Upsample to image size
    h, w = image.shape[2:]
    attention = F.interpolate(attention, size=(h, w), mode="bilinear", align_corners=False)
    attention = attention.squeeze().cpu().numpy()

    # Normalize
    if attention.max() > 0:
        attention = (attention - attention.min()) / (attention.max() - attention.min())

    return attention


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image: Original image (H, W, 3) in [0, 255] uint8.
        heatmap: Heatmap (H, W) in [0, 1].
        alpha: Overlay transparency.
        colormap: OpenCV colormap.

    Returns:
        Overlaid image (H, W, 3) uint8.
    """
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def generate_gradcam_samples(
    model: torch.nn.Module,
    test_loader: "DataLoader",
    model_name: str,
    output_dir: str | Path,
    device: str = "cuda",
    samples_per_class: int = 5,
) -> None:
    """Generate Grad-CAM visualizations for sample test images.

    Creates a grid: 5 severity grades × N samples with heatmap overlays.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        model_name: Model identifier.
        output_dir: Where to save visualizations.
        device: Device for inference.
        samples_per_class: Number of samples per severity grade.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine visualization method
    is_vit = "dinov2" in model_name or "vit" in model_name

    if not is_vit:
        target_layer = get_target_layer(model, model_name)
        gradcam = GradCAM(model, target_layer)

    # Collect samples per class
    class_samples: dict[int, list] = {i: [] for i in range(5)}

    model.eval()
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            label_int = label.item()
            if len(class_samples[label_int]) < samples_per_class:
                class_samples[label_int].append(img)

        # Check if we have enough
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    # Generate visualizations
    fig, axes = plt.subplots(5, samples_per_class, figsize=(3 * samples_per_class, 15))

    for class_idx in range(5):
        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]

            if sample_idx < len(class_samples[class_idx]):
                img_tensor = class_samples[class_idx][sample_idx].unsqueeze(0).to(device)

                # Generate heatmap
                if is_vit:
                    heatmap = generate_attention_map(model, img_tensor)
                else:
                    heatmap = gradcam.generate(img_tensor, target_class=class_idx)

                # Denormalize image for display
                img_display = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
                img_display = (img_display * np.array([0.229, 0.224, 0.225]) +
                              np.array([0.485, 0.456, 0.406]))
                img_display = np.clip(img_display * 255, 0, 255).astype(np.uint8)

                # Resize heatmap
                heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))

                # Overlay
                overlay = overlay_heatmap(img_display, heatmap_resized)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                ax.imshow(overlay)

                # Get prediction
                with torch.no_grad():
                    output = model(img_tensor)
                    if isinstance(output, dict):
                        output = output["logits"]
                    pred = output.argmax(dim=1).item()
                    conf = F.softmax(output, dim=-1)[0, pred].item()

                ax.set_title(f"Pred: {pred} ({conf:.2f})", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

            ax.axis("off")
            if sample_idx == 0:
                ax.set_ylabel(CLASS_NAMES[class_idx], fontsize=10)

    method = "Attention Map" if is_vit else "Grad-CAM"
    plt.suptitle(f"{method} Visualizations — {model_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_gradcam_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not is_vit:
        gradcam.cleanup()

    logger.info("Grad-CAM samples saved to %s", output_dir)
