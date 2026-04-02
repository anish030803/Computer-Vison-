"""FastAPI inference server for DR detection.

Endpoints:
  POST /api/predict — Predict DR severity from fundus image
  GET  /health      — Health check
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from server.model_loader import CLASS_NAMES, ModelManager
from server.schemas import (
    GradCAMResult,
    HealthResponse,
    InferenceMetadata,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from src.data.preprocessing import ben_graham_preprocess

logger = logging.getLogger("dr_detection")

# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    config_path = os.environ.get("MODEL_CONFIG", "configs/train_efficientnet.yaml")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)

    try:
        model_manager.load(config_path, checkpoint_path)
        logger.info("Server ready")
    except Exception as e:
        logger.error("Failed to load model: %s", e)

    yield

    logger.info("Server shutting down")


app = FastAPI(
    title="DR Detection API",
    description="Automated Diabetic Retinopathy Detection & Grading",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        model_name=model_manager.model_name,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict DR severity from a base64-encoded fundus image."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Decode image
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Preprocess
    image_size = model_manager.image_size
    preprocessed = ben_graham_preprocess(image_np, target_size=image_size)

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (preprocessed - mean) / std

    # To tensor
    tensor = (
        torch.from_numpy(normalized.transpose(2, 0, 1))
        .float()
        .unsqueeze(0)
        .to(model_manager.device)
    )

    # Predict
    with torch.no_grad():
        output = model_manager.model(tensor)
        if isinstance(output, dict):
            output = output["logits"]
        probs = F.softmax(output, dim=-1)[0].cpu().numpy()

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    # Build prediction result
    prediction = PredictionResult(
        **{
            "class": pred_class,
            "label": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        }
    )

    # Grad-CAM (optional)
    gradcam_result = None
    if request.return_gradcam:
        try:
            gradcam_result = _generate_gradcam(tensor, pred_class, image_np, image_size)
        except Exception as e:
            logger.warning("Grad-CAM generation failed: %s", e)

    inference_time = (time.time() - start_time) * 1000

    return PredictionResponse(
        prediction=prediction,
        gradcam=gradcam_result,
        metadata=InferenceMetadata(
            model=model_manager.model_name or "unknown",
            inference_time_ms=round(inference_time, 1),
            preprocessing_applied=f"ben_graham_{image_size}",
        ),
    )


def _generate_gradcam(
    tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray,
    image_size: int,
) -> GradCAMResult:
    """Generate Grad-CAM overlay."""
    from src.evaluation.gradcam import GradCAM, get_target_layer, overlay_heatmap

    model_name = model_manager.model_name or ""

    if "dinov2" in model_name:
        from src.evaluation.gradcam import generate_attention_map
        heatmap = generate_attention_map(model_manager.model, tensor)
    else:
        target_layer = get_target_layer(model_manager.model, model_name)
        gradcam = GradCAM(model_manager.model, target_layer)
        heatmap = gradcam.generate(tensor, target_class)
        gradcam.cleanup()

    # Overlay on resized original image
    img_resized = cv2.resize(original_image, (image_size, image_size))
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
    overlay = overlay_heatmap(img_resized, heatmap_resized)

    # Encode to base64
    _, buffer = cv2.imencode(".png", overlay)
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return GradCAMResult(
        heatmap=heatmap_b64,
        attention_regions=[],
    )
