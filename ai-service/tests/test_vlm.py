"""
Unit tests for VLM report generation, grounding, PDF report generation, DICOM ingestion, and async job store.
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

from dicom.loader import validate_dicom_bytes, _to_rgb_uint8 as to_rgb_uint8
from jobs.store import AsyncJobStore, JobStatus, JobRecord
from vlm.pdf_export import generate_pdf_report
from vlm.grounding import ReportGroundingChecker
from vlm.generator import GroundedReportGenerator
from vlm.templates import TEMPLATES, NORMAL_TEMPLATE


def make_synthetic_dicom_bytes() -> bytes:
    """Creates minimum valid DICOM preamble (128 zero bytes + DICM magic)."""
    preamble = b"\x00" * 128
    magic = b"DICM"
    return preamble + magic + b"\x00" * 256


def make_jpeg_b64(width: int = 64, height: int = 64) -> str:
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 200)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestDicomLoader:
    def test_validate_dicom_magic_bytes_valid(self):
        valid_bytes = make_synthetic_dicom_bytes()
        assert validate_dicom_bytes(valid_bytes) is True

    def test_validate_dicom_magic_bytes_invalid(self):
        invalid_bytes = b"NOT_A_DICOM_FILE_HEADER" + b"\x00" * 120
        assert validate_dicom_bytes(invalid_bytes) is False

    def test_validate_dicom_magic_bytes_too_short(self):
        too_short = b"\x00" * 50
        assert validate_dicom_bytes(too_short) is False

    def test_to_rgb_uint8_grayscale(self):
        gray = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        rgb = to_rgb_uint8(gray)
        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8
        assert rgb[0, 0, 0] == 0
        assert rgb[1, 0, 0] == 255

    def test_to_rgb_uint8_clipping(self):
        clipped = np.array([[-1.0, 2.0]], dtype=np.float32)
        rgb = to_rgb_uint8(clipped)
        assert rgb.min() >= 0
        assert rgb.max() <= 255


class TestAsyncJobStore:
    @pytest.fixture
    def store(self):
        return AsyncJobStore()

    async def test_create_job_returns_uuid(self, store):
        job_id = await store.create_job(metadata={"test": "val"})
        assert isinstance(job_id, str)
        assert len(job_id) == 36
        job = await store.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.PENDING

    async def test_job_lifecycle(self, store):
        job_id = await store.create_job()
        await store.mark_running(job_id)
        job = await store.get_job(job_id)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

        result_payload = {"predictions": [], "metadata": {}}
        await store.mark_completed(job_id, result_payload)
        job = await store.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result == result_payload
        assert job.completed_at is not None

    async def test_job_failure_lifecycle(self, store):
        job_id = await store.create_job()
        await store.mark_running(job_id)
        await store.mark_failed(job_id, "CUDA out of memory")
        job = await store.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "CUDA out of memory" in job.error

    async def test_list_jobs(self, store):
        for _ in range(5):
            await store.create_job()
        jobs = await store.list_jobs(limit=3)
        assert len(jobs) == 3

    async def test_cleanup_old_jobs(self, store):
        job_id = await store.create_job()
        await store.mark_completed(job_id, {})
        job = await store.get_job(job_id)
        job.submitted_at = time.time() - 7200
        job.completed_at = time.time() - 7200

        deleted = await store.cleanup_old_jobs(max_age_seconds=3600)
        assert deleted >= 1
        assert await store.get_job(job_id) is None

    async def test_missing_job_returns_none(self, store):
        job = await store.get_job("non-existent-uuid")
        assert job is None


class TestPDFReportGenerator:
    @pytest.fixture
    def dummy_predictions(self):
        return [
            {"disease": "Atelectasis", "probability": 0.72, "threshold": 0.5, "detected": True, "uncertainty": 0.04, "is_uncertain": False},
            {"disease": "Cardiomegaly", "probability": 0.12, "threshold": 0.5, "detected": False, "uncertainty": 0.01, "is_uncertain": False},
            {"disease": "Effusion", "probability": 0.55, "threshold": 0.5, "detected": True, "uncertainty": 0.08, "is_uncertain": False},
        ]

    @pytest.fixture
    def dummy_report(self):
        return {
            "status": "completed",
            "findings": ["Plate-like opacity consistent with atelectasis.", "Blunting of the costophrenic angle."],
            "impression": ["Atelectasis and pleural effusion identified."],
            "detected_conditions": ["Atelectasis", "Effusion"],
        }

    def test_pdf_generates_non_empty_bytes(self, dummy_predictions, dummy_report):
        pdf_bytes = generate_pdf_report(
            predictions=dummy_predictions,
            report_dict=dummy_report,
            patient_id="TEST-PATIENT-001",
            request_id="req-test-1234",
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_with_uncertain_predictions(self, dummy_predictions, dummy_report):
        dummy_predictions[0]["is_uncertain"] = True
        dummy_predictions[0]["uncertainty"] = 0.22
        pdf_bytes = generate_pdf_report(
            predictions=dummy_predictions,
            report_dict=dummy_report,
            patient_id="UNCERTAIN-002",
        )
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_normal_scan(self):
        predictions = [
            {"disease": d, "probability": 0.05, "threshold": 0.5, "detected": False, "uncertainty": 0.01}
            for d in ["Atelectasis", "Cardiomegaly", "Effusion"]
        ]
        report = {
            "status": "completed",
            "findings": [NORMAL_TEMPLATE["finding"]],
            "impression": [NORMAL_TEMPLATE["impression"]],
            "detected_conditions": [],
        }
        pdf_bytes = generate_pdf_report(predictions=predictions, report_dict=report)
        assert pdf_bytes[:4] == b"%PDF"


class TestVLMFindingsParser:
    def test_parse_findings_list_format(self):
        checker = ReportGroundingChecker(target_classes=["Atelectasis", "Effusion"])
        is_grounded, unsupported = checker.validate_report(
            detected_diseases=["Atelectasis"],
            generated_text="Atelectasis identified in the lower right lobe. No effusion seen.",
        )
        assert is_grounded is True
        assert len(unsupported) == 0

    def test_evaluate_detects_hallucination(self):
        checker = ReportGroundingChecker(target_classes=["Atelectasis", "Effusion", "Pneumothorax"])
        is_grounded, unsupported = checker.validate_report(
            detected_diseases=["Atelectasis"],
            generated_text="Atelectasis in the lower lobe with extensive pneumothorax.",
        )
        assert is_grounded is False
        assert "Pneumothorax" in unsupported

    def test_grounded_report_generator_normal(self):
        gen = GroundedReportGenerator(target_classes=["Atelectasis", "Cardiomegaly"])
        report = gen.generate_report([
            {"disease": "Atelectasis", "probability": 0.10, "detected": False},
            {"disease": "Cardiomegaly", "probability": 0.05, "detected": False},
        ])
        assert report["status"] == "completed"
        assert len(report["detected_conditions"]) == 0
        assert NORMAL_TEMPLATE["finding"] in report["findings"]
