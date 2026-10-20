"""
Pydantic schemas for the AI service supporting Multi-Disease screening and Report Generation.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded chest X-ray image (JPEG/PNG/WebP).",
        min_length=100,
    )
    mime_type: str = Field(
        default="image/jpeg",
        description="Image MIME type.",
    )
    use_ensemble: bool = Field(
        default=False,
        description="Whether to run multi-model ensemble (DenseNet-121, ResNet-50, Swin-T).",
    )
    use_tta: bool = Field(
        default=False,
        description="Whether to execute Test-Time Augmentation (TTA).",
    )
    tta_steps: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of TTA augmentation passes (1=Identity, 2=HFlip, 3=CenterScale, 4=Contrast).",
    )

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/dicom"}
        if v not in allowed:
            raise ValueError(f"mime_type must be one of {allowed}")
        return v


# Backward compatible single prediction response
class PredictionResponse(BaseModel):
    prediction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="TB probability score (0–1). >0.5 indicates TB positive.",
    )
    heatmap: str = Field(
        ...,
        description="Base64-encoded JPEG Grad-CAM heatmap.",
    )
    overlay: str = Field(
        ...,
        description="Base64-encoded JPEG Grad-CAM overlay on original image.",
    )
    label: str = Field(
        default="",
        description="Human-readable label: 'TB Positive' or 'TB Negative'.",
    )
    confidence: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Confidence percentage.",
    )

    def model_post_init(self, __context) -> None:
        self.label = "TB Positive" if self.prediction > 0.5 else "TB Negative"
        self.confidence = round(max(self.prediction, 1 - self.prediction) * 100)


# Multi-Disease Analysis Schemas
class DiseasePrediction(BaseModel):
    disease: str
    probability: float
    threshold: float
    detected: bool
    uncertainty: Optional[float] = Field(
        default=None,
        description="Predictive standard deviation across ensemble / TTA passes.",
    )
    confidence_interval: Optional[List[float]] = Field(
        default=None,
        description="95% confidence interval [lower_bound, upper_bound].",
    )
    is_uncertain: Optional[bool] = Field(
        default=None,
        description="Flag indicating high predictive uncertainty requiring clinical attention.",
    )


class HeatmapItem(BaseModel):
    disease: str
    heatmap: str  # Base64 JPEG
    overlay: str  # Base64 JPEG


class DetectedConditionInfo(BaseModel):
    name: str
    confidence: float


class ModelConfidenceInfo(BaseModel):
    overall: float
    per_disease: Dict[str, float]


class ReportDraft(BaseModel):
    status: str = "draft"
    findings: List[str]
    impression: List[str]
    detected_conditions: List[DetectedConditionInfo]
    model_confidence: ModelConfidenceInfo
    review_required: bool = True
    unsupported_findings: List[str] = []
    disclaimer: str = "AI-generated draft for research and educational purposes. Human review is required."


class MultiDiseaseAnalyzeResponse(BaseModel):
    request_id: str
    predictions: List[DiseasePrediction]
    heatmaps: List[HeatmapItem]
    report: ReportDraft
    metadata: Dict[str, Any]


class ReportValidationRequest(BaseModel):
    predictions: List[DiseasePrediction]
    generated_text: str


class ReportValidationResponse(BaseModel):
    is_grounded: bool
    unsupported_claims: List[str]
    status: str
