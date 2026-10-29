import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

"""
NIH ChestX-ray14 Multi-Disease Chest X-Ray Screening & Report Generation AI Service.
FastAPI Application — Phase 2 upgrade:
  - TorchXRayVision DenseNet-121 + ResNet-50 ensemble, TTA, and uncertainty
  - DICOM file upload endpoint
  - Async background job queue (submit → poll → retrieve)
  - ACR-style PDF report download endpoint
"""
import asyncio
import base64
import io
import uuid
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from models.tb_model import TBModel
from src.models.ensemble import MultiModelEnsemble
from src.jobs.store import AsyncJobStore, JobStatus, init_job_store, get_job_store
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

print("\n--- NIH ChestX-ray14 MULTI-DISEASE AI SERVICE BOOTING UP (Phase 2) ---")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-service")

model: TBModel | None = None
ensemble: MultiModelEnsemble | None = None
vlm_generator: VLMReportGenerator | None = None
grounding_checker: ReportGroundingChecker | None = None
job_store: Optional[AsyncJobStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, ensemble, VLM, and job store on startup."""
    global model, ensemble, vlm_generator, grounding_checker, job_store
    logger.info("Initializing Multi-Disease Screening Model...")
    try:
        model = TBModel()
        model.load()
        logger.info(f"Model loaded with {model.num_classes} target classes.")

        # Initialize ensemble with the TorchXRayVision DenseNet checkpoint.
        ensemble = MultiModelEnsemble(num_classes=model.num_classes, classes=model.classes)
        ensemble.register_model("densenet121", model.net)
        # Explicitly initialize the requested XRV ResNet-50 weights as the second model.
        from src.models.resnet import NIHResNet50
        ensemble.register_model("resnet50", NIHResNet50(num_classes=model.num_classes))

        vlm_generator = VLMReportGenerator(target_classes=model.classes)
        vlm_generator.load_vlm()

        grounding_checker = ReportGroundingChecker(target_classes=model.classes)

        # Initialize async job store
        job_store = init_job_store()
        logger.info("Async job store initialized.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        model = None
        ensemble = None
    yield
    logger.info("Shutting down AI service.")
    model = None
    ensemble = None
    vlm_generator = None


app = FastAPI(
    title="NIH ChestX-ray14 Multi-Disease Screening API",
    description=(
        "14-Class Multi-Label Classification (NIH ChestX-ray14), "
        "Per-Disease Grad-CAM, Grounded Radiology Report Generation, "
        "DICOM ingestion, Async Job Queue, and ACR-style PDF Reports."
    ),
    version="2.2.0",
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


# ─── Health & Info Endpoints ──────────────────────────────────────────────────

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
        "model_version": "2.2.0-nih14class-phase2",
        "num_classes": model.num_classes,
        "device": str(model.device),
        "job_store_active": job_store is not None,
    }


@app.get("/api/v1/model/info", tags=["monitoring"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "model_architecture": "DenseNet-121 NIH ChestX-ray14 (14-class multi-label)",
        "model_version": "2.2.0-nih14class-phase2",
        "dataset": "NIH ChestX-ray14",
        "num_classes": model.num_classes,
        "classes": model.classes,
        "thresholds": model.thresholds,
        "device": str(model.device),
    }


# ─── Core Analysis Endpoint ───────────────────────────────────────────────────

async def _run_analysis(
    image_b64: str,
    mime_type: str,
    use_ensemble: bool,
    use_tta: bool,
    tta_steps: int,
    req_id: str,
) -> dict:
    """Shared analysis logic used by both /analyze and async job runner."""
    start_time = time.time()
    image_tensor, original_np = preprocess_image(image_b64, mime_type)

    if (use_ensemble or use_tta) and ensemble is not None:
        predictions_raw, feat_maps, ensemble_meta = ensemble.predict_multi_structured(
            image_tensor,
            thresholds=model.thresholds,
            use_tta=use_tta,
            tta_steps=tta_steps,
        )
        grad_dict = getattr(model.net, "_gradients", {})
    else:
        predictions_raw, feat_maps, grad_dict = model.predict_multi(image_tensor)
        ensemble_meta = {
            "ensemble_enabled": False,
            "tta_enabled": False,
            "models_evaluated": ["densenet121"],
            "total_passes": 1,
        }

    detected_classes = [p["disease"] for p in predictions_raw if p["detected"]]
    if not detected_classes and predictions_raw:
        top_disease = max(predictions_raw, key=lambda x: x["probability"])["disease"]
        detected_classes = [top_disease]

    multi_cam_dict = generate_multi_gradcam(original_np, feat_maps, grad_dict, model.classes)
    heatmaps_list = []
    for disease, (hm_b64, ov_b64) in multi_cam_dict.items():
        if disease in detected_classes or len(heatmaps_list) < 3:
            heatmaps_list.append({"disease": disease, "heatmap": hm_b64, "overlay": ov_b64})

    pil_img = Image.fromarray(original_np)
    report_dict = vlm_generator.generate_report(pil_img, predictions_raw)
    proc_time = round((time.time() - start_time) * 1000, 2)

    return {
        "request_id": req_id,
        "predictions": predictions_raw,
        "heatmaps": heatmaps_list,
        "report": report_dict,
        "metadata": {
            "model_version": "2.2.0-nih14class-phase2",
            "dataset": "NIH ChestX-ray14",
            "processing_time_ms": proc_time,
            "target_classes_count": len(model.classes),
            "ensemble_enabled": ensemble_meta.get("ensemble_enabled", use_ensemble),
            "tta_enabled": ensemble_meta.get("tta_enabled", use_tta),
            "models_evaluated": ensemble_meta.get("models_evaluated", ["densenet121"]),
            "total_passes": ensemble_meta.get("total_passes", 1),
            "mean_uncertainty": ensemble_meta.get("mean_uncertainty"),
        },
    }


@app.post("/api/v1/analyze", response_model=MultiDiseaseAnalyzeResponse, tags=["inference"])
async def analyze_xray(request: PredictionRequest):
    """Synchronous multi-label classification + Grad-CAM + grounded report."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    req_id = f"req-{uuid.uuid4().hex[:8]}"
    try:
        result = await _run_analysis(
            request.image, request.mime_type,
            request.use_ensemble, request.use_tta, request.tta_steps,
            req_id,
        )
        predictions_obj = [DiseasePrediction(**p) for p in result["predictions"]]
        heatmaps_obj = [HeatmapItem(**h) for h in result["heatmaps"]]
        report_obj = ReportDraft(**result["report"])
        logger.info(f"Analysis [{req_id}] complete in {result['metadata']['processing_time_ms']}ms.")
        return MultiDiseaseAnalyzeResponse(
            request_id=result["request_id"],
            predictions=predictions_obj,
            heatmaps=heatmaps_obj,
            report=report_obj,
            metadata=result["metadata"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Analysis error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


# ─── DICOM Upload Endpoint ────────────────────────────────────────────────────

@app.post("/api/v1/dicom/upload", tags=["dicom"])
async def upload_dicom(file: UploadFile = File(...)):
    """
    Upload a DICOM file (.dcm) and receive its extracted pixel data as base64 JPEG
    plus DICOM metadata. The returned base64 can be passed directly to /api/v1/analyze.
    """
    try:
        from src.dicom.loader import dicom_to_base64_jpeg, validate_dicom_bytes
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"DICOM support not available: {e}")

    if not file.filename or not file.filename.lower().endswith((".dcm", ".dicom")):
        # Also accept application/dicom content type
        if file.content_type not in ("application/dicom", "application/octet-stream"):
            logger.warning("DICOM upload: suspicious filename '%s' content-type '%s'", file.filename, file.content_type)

    raw = await file.read()
    if len(raw) < 132:
        raise HTTPException(status_code=422, detail="File too small to be a valid DICOM.")

    if not validate_dicom_bytes(raw):
        logger.warning("DICOM magic bytes missing — attempting pydicom force-read anyway.")

    try:
        b64_jpeg, metadata = dicom_to_base64_jpeg(raw)
    except ImportError as ie:
        raise HTTPException(status_code=500, detail=str(ie))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"DICOM parsing failed: {e}")

    return {
        "status": "ok",
        "image_b64": b64_jpeg,
        "mime_type": "image/jpeg",
        "dicom_metadata": metadata,
        "message": "DICOM converted successfully. Pass image_b64 and mime_type to /api/v1/analyze.",
    }


# ─── Async Job Submission Endpoints ──────────────────────────────────────────

async def _background_analysis_job(job_id: str, image_b64: str, mime_type: str,
                                    use_ensemble: bool, use_tta: bool, tta_steps: int):
    """Background coroutine that runs analysis and updates the job store."""
    store = get_job_store()
    await store.mark_running(job_id)
    try:
        result = await _run_analysis(image_b64, mime_type, use_ensemble, use_tta, tta_steps, job_id)
        await store.mark_completed(job_id, result)
    except Exception as exc:
        await store.mark_failed(job_id, str(exc))


@app.post("/api/v1/jobs/submit", tags=["async-jobs"])
async def submit_job(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Submit an analysis job for background processing.
    Returns a job_id immediately. Poll /api/v1/jobs/{job_id} for status.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if job_store is None:
        raise HTTPException(status_code=503, detail="Job store not initialized.")

    job_id = await job_store.create_job(metadata={
        "mime_type": request.mime_type,
        "use_ensemble": request.use_ensemble,
        "use_tta": request.use_tta,
        "tta_steps": request.tta_steps,
    })

    background_tasks.add_task(
        _background_analysis_job,
        job_id, request.image, request.mime_type,
        request.use_ensemble, request.use_tta, request.tta_steps,
    )

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING.value,
        "message": f"Job submitted. Poll GET /api/v1/jobs/{job_id} for status.",
    }


@app.get("/api/v1/jobs/{job_id}", tags=["async-jobs"])
async def get_job_status(job_id: str):
    """Poll the status and result of a submitted analysis job."""
    if job_store is None:
        raise HTTPException(status_code=503, detail="Job store not initialized.")
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()


@app.get("/api/v1/jobs", tags=["async-jobs"])
async def list_jobs(limit: int = 20):
    """List recent jobs (newest first)."""
    if job_store is None:
        raise HTTPException(status_code=503, detail="Job store not initialized.")
    return {"jobs": await job_store.list_jobs(limit=limit)}


# ─── PDF Report Download Endpoint ────────────────────────────────────────────

@app.post("/api/v1/report/pdf", tags=["report"])
async def generate_pdf(request: PredictionRequest):
    """
    Run analysis and return the result as an ACR-style PDF report download.
    Returns a PDF binary with Content-Disposition: attachment.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        from report_generation.pdf_report import generate_pdf_report
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"PDF generation not available: {e}")

    req_id = f"pdf-{uuid.uuid4().hex[:8]}"
    try:
        result = await _run_analysis(
            request.image, request.mime_type,
            request.use_ensemble, request.use_tta, request.tta_steps,
            req_id,
        )

        # Use the first heatmap overlay for PDF embedding
        hm_b64 = result["heatmaps"][0]["overlay"] if result["heatmaps"] else None

        pdf_bytes = generate_pdf_report(
            predictions=result["predictions"],
            report_dict=result["report"],
            patient_id="ANONYMOUS",
            heatmap_b64=hm_b64,
            ensemble_meta=result["metadata"],
            request_id=req_id,
        )
        logger.info(f"PDF report generated [{req_id}], size={len(pdf_bytes)} bytes")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="nih_report_{req_id}.pdf"',
                "X-Request-ID": req_id,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"PDF generation error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(exc)}")


# ─── PDF from Job Result ──────────────────────────────────────────────────────

@app.get("/api/v1/jobs/{job_id}/pdf", tags=["async-jobs", "report"])
async def get_job_pdf(job_id: str):
    """Download the PDF report for a completed job."""
    if job_store is None:
        raise HTTPException(status_code=503, detail="Job store not initialized.")

    try:
        from report_generation.pdf_report import generate_pdf_report
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"PDF generation not available: {e}")

    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job is not completed yet. Status: {job.status.value}")

    result = job.result
    hm_b64 = result["heatmaps"][0]["overlay"] if result.get("heatmaps") else None
    pdf_bytes = generate_pdf_report(
        predictions=result["predictions"],
        report_dict=result["report"],
        patient_id=job.metadata.get("patient_id", "ANONYMOUS"),
        ensemble_meta=result.get("metadata"),
        heatmap_b64=hm_b64,
        request_id=job_id,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="nih_report_{job_id}.pdf"'},
    )


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
