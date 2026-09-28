import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

"""
NIH ChestX-ray14 Multi-Disease Chest X-Ray Screening & Report Generation AI Service.
FastAPI Application with DenseNet-121 (14-class NIH model) & VLM/Grounded Reporting.
"""
import uuid
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.tb_model import TBModel
from schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    MultiDiseaseAnalyzeResponse,
    DiseasePrediction,
    HeatmapItem,
    ReportDraft,
    ReportValidationRequest,
    ReportValidationResponse,
)
from utils.image_processor import preprocess_image
from utils.gradcam import generate_single_gradcam, generate_multi_gradcam
from report_generation.vlm_report import VLMReportGenerator
from report_generation.report_grounding import ReportGroundingChecker

print("\n--- NIH ChestX-ray14 MULTI-DISEASE AI SERVICE BOOTING UP ---")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-service")

model: TBModel | None = None
vlm_generator: VLMReportGenerator | None = None
grounding_checker: ReportGroundingChecker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and VLM on startup."""
    global model, vlm_generator, grounding_checker
    logger.info("Initializing Multi-Disease Screening Model...")
    try:
        model = TBModel()
        model.load()
        logger.info(f"Model loaded with {model.num_classes} target classes.")

        vlm_generator = VLMReportGenerator(target_classes=model.classes)
        # Attempt loading VLM (falls back gracefully to Stage A grounded generator)
        vlm_generator.load_vlm()

        grounding_checker = ReportGroundingChecker(target_classes=model.classes)
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        model = None
    yield
    logger.info("Shutting down AI service.")
    model = None
    vlm_generator = None


app = FastAPI(
    title="NIH ChestX-ray14 Multi-Disease Screening API",
    description="14-Class Multi-Label Classification (NIH ChestX-ray14), Per-Disease Grad-CAM, and Grounded Radiology Report Generation",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


@app.get("/health", tags=["monitoring"])
@app.get("/api/v1/health", tags=["monitoring"])
async def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False},
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "2.1.0-nih14class",
        "num_classes": model.num_classes,
        "device": str(model.device),
    }


@app.get("/api/v1/model/info", tags=["monitoring"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "model_architecture": "DenseNet-121 NIH ChestX-ray14 (14-class multi-label)",
        "model_version": "2.1.0-nih14class",
        "dataset": "NIH ChestX-ray14",
        "num_classes": model.num_classes,
        "classes": model.classes,
        "thresholds": model.thresholds,
        "device": str(model.device),
    }


# ─── Multi-Disease Analysis Endpoint ──────────────────────────────────────────
@app.post("/api/v1/analyze", response_model=MultiDiseaseAnalyzeResponse, tags=["inference"])
async def analyze_xray(request: PredictionRequest):
    """
    Runs multi-label classification, per-disease Grad-CAM, and grounded report generation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    start_time = time.time()
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        # 1. Preprocess Image
        image_tensor, original_np = preprocess_image(request.image, request.mime_type)

        # 2. Multi-label inference
        predictions_raw, feat_maps, grad_dict = model.predict_multi(image_tensor)

        # 3. Filter detected conditions for Grad-CAM generation
        detected_classes = [p["disease"] for p in predictions_raw if p["detected"]]
        if not detected_classes and predictions_raw:
            # If no threshold exceeded, generate heatmap for highest confidence disease
            top_disease = max(predictions_raw, key=lambda x: x["probability"])["disease"]
            detected_classes = [top_disease]

        # 4. Generate Multi-GradCAM
        multi_cam_dict = generate_multi_gradcam(
            original_np, feat_maps, grad_dict, model.classes
        )

        heatmaps_list = []
        for disease, (hm_b64, ov_b64) in multi_cam_dict.items():
            if disease in detected_classes or len(heatmaps_list) < 3:
                heatmaps_list.append(HeatmapItem(disease=disease, heatmap=hm_b64, overlay=ov_b64))

        # 5. Generate Grounded Draft Report
        from PIL import Image
        pil_img = Image.fromarray(original_np)
        report_dict = vlm_generator.generate_report(pil_img, predictions_raw)

        proc_time = round((time.time() - start_time) * 1000, 2)

        predictions_obj = [DiseasePrediction(**p) for p in predictions_raw]
        report_obj = ReportDraft(**report_dict)

        logger.info(f"Analysis [{req_id}] complete in {proc_time}ms.")

        return MultiDiseaseAnalyzeResponse(
            request_id=req_id,
            predictions=predictions_obj,
            heatmaps=heatmaps_list,
            report=report_obj,
            metadata={
                "model_version": "2.1.0-nih14class",
                "dataset": "NIH ChestX-ray14",
                "processing_time_ms": proc_time,
                "target_classes_count": len(model.classes),
            },
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Analysis error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


# ─── Report Validation / Hallucination Endpoint ──────────────────────────────
@app.post("/api/v1/report/validate", response_model=ReportValidationResponse, tags=["report"])
async def validate_report(request: ReportValidationRequest):
    if grounding_checker is None:
        raise HTTPException(status_code=503, detail="Checker not initialized.")

    detected = [p.disease for p in request.predictions if p.detected]
    is_grounded, unsupported = grounding_checker.validate_report(detected, request.generated_text)

    return ReportValidationResponse(
        is_grounded=is_grounded,
        unsupported_claims=unsupported,
        status="grounded" if is_grounded else "needs_review",
    )


# ─── Backward Compatible Single-Prediction Endpoint ──────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict_legacy(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        image_tensor, original_np = preprocess_image(request.image, request.mime_type)
        score, feat_maps, gradients = model.predict_with_gradients(image_tensor)
        heatmap_b64, overlay_b64 = generate_single_gradcam(original_np, feat_maps, gradients)

        return PredictionResponse(
            prediction=round(float(score), 6),
            heatmap=heatmap_b64,
            overlay=overlay_b64,
        )
    except Exception as exc:
        logger.error(f"Legacy prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")
