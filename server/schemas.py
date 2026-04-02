"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for /api/predict endpoint."""

    image: str = Field(
        ...,
        description="Base64-encoded fundus image",
    )
    return_gradcam: bool = Field(
        default=False,
        description="Whether to generate and return Grad-CAM heatmap",
    )


class ProbabilityDistribution(BaseModel):
    """Per-class probability distribution."""

    no_dr: float = Field(..., alias="No DR")
    mild_npdr: float = Field(..., alias="Mild NPDR")
    moderate_npdr: float = Field(..., alias="Moderate NPDR")
    severe_npdr: float = Field(..., alias="Severe NPDR")
    proliferative_dr: float = Field(..., alias="Proliferative DR")

    model_config = {"populate_by_name": True}


class PredictionResult(BaseModel):
    """Classification prediction result."""

    class_index: int = Field(..., alias="class", description="Predicted class index (0-4)")
    label: str = Field(..., description="Human-readable severity label")
    confidence: float = Field(..., description="Confidence score for predicted class")
    probabilities: dict[str, float] = Field(
        ..., description="Per-class probability distribution"
    )

    model_config = {"populate_by_name": True}


class GradCAMResult(BaseModel):
    """Grad-CAM visualization result."""

    heatmap: str = Field(..., description="Base64-encoded heatmap overlay image")
    attention_regions: list[str] = Field(
        default_factory=list,
        description="Detected attention region descriptions",
    )


class InferenceMetadata(BaseModel):
    """Metadata about the inference request."""

    model: str = Field(..., description="Model name and version")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    preprocessing_applied: str = Field(..., description="Preprocessing method used")


class PredictionResponse(BaseModel):
    """Full response from /api/predict endpoint."""

    prediction: PredictionResult
    gradcam: Optional[GradCAMResult] = None
    metadata: InferenceMetadata


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = "ok"
    model_loaded: bool = False
    model_name: Optional[str] = None
