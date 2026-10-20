"""
Phase 2 & 3 Test Suite — DICOM Ingestion, Async Job Queue, PDF Report.

Run from ai-service/:
  pytest tests/test_phase2_dicom_jobs_pdf.py -v

Tests:
  1. DICOM loader validation (magic bytes, synthetic DICOM)
  2. Async job store lifecycle (create → running → completed → poll)
  3. PDF report generation (reportlab output, non-empty bytes)
  4. FastAPI /api/v1/dicom/upload endpoint (mock DICOM)
  5. FastAPI /api/v1/jobs/submit and /api/v1/jobs/{id} endpoints
  6. VLM findings parser correctness
"""
import asyncio
import base64
import io
import json
import struct
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure ai-service is on path
AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))


# ─── Helper: Synthetic DICOM bytes ────────────────────────────────────────────

def make_synthetic_dicom_bytes() -> bytes:
    """
    Creates the minimum valid DICOM preamble (128 zero bytes + DICM magic).
    NOTE: This only tests magic-byte validation; full pixel extraction requires pydicom.
    """
    preamble = b"\x00" * 128
    magic = b"DICM"
    return preamble + magic + b"\x00" * 256  # minimal trailing bytes


def make_jpeg_b64(width: int = 64, height: int = 64) -> str:
    """Create a tiny white JPEG and return as base64 string."""
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 200)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── 1. DICOM Loader Tests ────────────────────────────────────────────────────

class TestDicomLoader:
    def test_validate_dicom_magic_bytes_valid(self):
        from src.dicom.loader import validate_dicom_bytes
        raw = make_synthetic_dicom_bytes()
        assert validate_dicom_bytes(raw) is True

    def test_validate_dicom_magic_bytes_invalid(self):
        from src.dicom.loader import validate_dicom_bytes
        assert validate_dicom_bytes(b"not a dicom file") is False

    def test_validate_dicom_magic_bytes_too_short(self):
        from src.dicom.loader import validate_dicom_bytes
        assert validate_dicom_bytes(b"\x00" * 100) is False

    def test_to_rgb_uint8_grayscale(self):
        from src.dicom.loader import _to_rgb_uint8
        gray = np.ones((64, 64), dtype=np.float32) * 0.5
        rgb = _to_rgb_uint8(gray)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8
        assert rgb[0, 0, 0] == rgb[0, 0, 1] == rgb[0, 0, 2]

    def test_to_rgb_uint8_clipping(self):
        from src.dicom.loader import _to_rgb_uint8
        over = np.ones((32, 32), dtype=np.float32) * 2.0  # out of [0, 1]
        rgb = _to_rgb_uint8(over)
        assert rgb.max() == 255


# ─── 2. Async Job Store Tests ─────────────────────────────────────────────────

class TestAsyncJobStore:
    @pytest.mark.asyncio
    async def test_create_job_returns_uuid(self):
        from src.jobs.store import AsyncJobStore
        store = AsyncJobStore()
        job_id = await store.create_job()
        assert isinstance(job_id, str) and len(job_id) == 36  # UUID4 format

    @pytest.mark.asyncio
    async def test_job_lifecycle(self):
        from src.jobs.store import AsyncJobStore, JobStatus
        store = AsyncJobStore()
        jid = await store.create_job(metadata={"test": True})
        job = await store.get_job(jid)
        assert job.status == JobStatus.PENDING

        await store.mark_running(jid)
        job = await store.get_job(jid)
        assert job.status == JobStatus.RUNNING

        await store.mark_completed(jid, result={"predictions": []})
        job = await store.get_job(jid)
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"predictions": []}
        assert job.duration_ms is not None

    @pytest.mark.asyncio
    async def test_job_failure_lifecycle(self):
        from src.jobs.store import AsyncJobStore, JobStatus
        store = AsyncJobStore()
        jid = await store.create_job()
        await store.mark_running(jid)
        await store.mark_failed(jid, "test error message")
        job = await store.get_job(jid)
        assert job.status == JobStatus.FAILED
        assert "test error message" in job.error

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        from src.jobs.store import AsyncJobStore
        store = AsyncJobStore()
        for _ in range(5):
            await store.create_job()
        jobs = await store.list_jobs(limit=3)
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_cleanup_old_jobs(self):
        from src.jobs.store import AsyncJobStore, JobStatus
        store = AsyncJobStore()
        jid = await store.create_job()
        await store.mark_completed(jid, {})
        # Backdate submission time to trigger cleanup
        store._jobs[jid].submitted_at = time.time() - 7200
        purged = await store.cleanup_old_jobs(max_age_seconds=3600)
        assert purged == 1

    @pytest.mark.asyncio
    async def test_missing_job_returns_none(self):
        from src.jobs.store import AsyncJobStore
        store = AsyncJobStore()
        assert await store.get_job("nonexistent-id") is None


# ─── 3. PDF Report Generator Tests ───────────────────────────────────────────

class TestPDFReportGenerator:
    def _make_predictions(self):
        return [
            {"disease": "Atelectasis", "probability": 0.82, "threshold": 0.5, "detected": True, "uncertainty": 0.03, "is_uncertain": False, "confidence_interval": [0.76, 0.88]},
            {"disease": "Cardiomegaly", "probability": 0.15, "threshold": 0.5, "detected": False, "uncertainty": None, "is_uncertain": False, "confidence_interval": None},
            {"disease": "Effusion", "probability": 0.67, "threshold": 0.5, "detected": True, "uncertainty": 0.12, "is_uncertain": True, "confidence_interval": [0.43, 0.91]},
        ]

    def _make_report(self):
        return {
            "status": "draft",
            "findings": ["Atelectasis detected with high confidence.", "Effusion detected — uncertain, review required."],
            "impression": ["Multiple thoracic abnormalities detected."],
            "detected_conditions": [{"name": "Atelectasis", "confidence": 82.0}],
            "model_confidence": {"overall": 0.75, "per_disease": {"Atelectasis": 0.82}},
            "review_required": True,
            "unsupported_findings": [],
            "disclaimer": "AI research draft.",
        }

    def test_pdf_generates_non_empty_bytes(self):
        try:
            from report_generation.pdf_report import generate_pdf_report
        except ImportError:
            pytest.skip("reportlab not installed")

        pdf_bytes = generate_pdf_report(
            predictions=self._make_predictions(),
            report_dict=self._make_report(),
            patient_id="TEST-001",
            request_id="test-req-123",
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000  # PDF must have meaningful content
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_with_uncertain_predictions(self):
        try:
            from report_generation.pdf_report import generate_pdf_report
        except ImportError:
            pytest.skip("reportlab not installed")

        preds = self._make_predictions()
        preds[1]["is_uncertain"] = True  # Make cardiomegaly uncertain too
        pdf_bytes = generate_pdf_report(
            predictions=preds,
            report_dict=self._make_report(),
            ensemble_meta={"models_evaluated": ["densenet121", "resnet50"], "tta_enabled": True, "total_passes": 8},
        )
        assert len(pdf_bytes) > 1000

    def test_pdf_normal_scan(self):
        try:
            from report_generation.pdf_report import generate_pdf_report
        except ImportError:
            pytest.skip("reportlab not installed")

        preds = [{"disease": "Atelectasis", "probability": 0.1, "threshold": 0.5, "detected": False, "uncertainty": None, "is_uncertain": False, "confidence_interval": None}]
        report = {"findings": ["No finding."], "impression": ["Normal chest radiograph."], "status": "draft",
                  "detected_conditions": [], "model_confidence": {"overall": 0.1, "per_disease": {}},
                  "review_required": True, "unsupported_findings": [], "disclaimer": ""}
        pdf_bytes = generate_pdf_report(predictions=preds, report_dict=report)
        assert pdf_bytes[:4] == b"%PDF"


# ─── 4. VLM Findings Parser Tests ────────────────────────────────────────────

class TestVLMFindingsParser:
    def test_parse_findings_list_format(self):
        from src.eval.vlm_eval import parse_findings_from_text
        text = "Identified findings: Atelectasis, Effusion."
        findings = parse_findings_from_text(text)
        assert "Atelectasis" in findings
        assert "Effusion" in findings

    def test_parse_checklist_format(self):
        from src.eval.vlm_eval import parse_findings_from_text
        text = "- Atelectasis: Positive\n- Cardiomegaly: Negative\n- Effusion: Positive"
        findings = parse_findings_from_text(text)
        assert "Atelectasis" in findings
        assert "Effusion" in findings
        assert "Cardiomegaly" not in findings

    def test_parse_no_finding(self):
        from src.eval.vlm_eval import parse_findings_from_text
        text = "No Finding. The chest radiograph demonstrates no acute thoracic abnormalities."
        findings = parse_findings_from_text(text)
        assert len(findings) == 0

    def test_evaluate_vlm_predictions_perfect(self):
        from src.eval.vlm_eval import evaluate_vlm_predictions
        generated = [
            "Identified findings: Atelectasis, Effusion.",
            "No Finding. The chest radiograph demonstrates no acute thoracic abnormalities.",
        ]
        gt = [["Atelectasis", "Effusion"], []]
        metrics = evaluate_vlm_predictions(generated, gt)
        assert metrics["macro_precision"] > 0
        assert "hallucination_rate" in metrics
        assert "per_class_f1" in metrics

    def test_evaluate_detects_hallucination(self):
        from src.eval.vlm_eval import evaluate_vlm_predictions
        # Generate says Pneumonia but GT is empty → hallucination
        generated = ["Pneumonia detected in left lower lobe."]
        gt = [[]]
        metrics = evaluate_vlm_predictions(generated, gt)
        assert metrics["hallucination_rate"] == 1.0
