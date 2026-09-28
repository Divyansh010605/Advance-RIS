"""
Integration test: Backend → AI Service → Full response pipeline.
Requires both services running (or use Docker Compose).

Run with:
  BACKEND_URL=http://localhost:5000  pytest tests/integration/test_integration.py -v
"""
import base64
import os
import sys

import httpx
import numpy as np
import pytest
from io import BytesIO
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
AI_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
TIMEOUT = 60.0


def make_dummy_b64_jpeg(width: int = 224, height: int = 224) -> str:
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def auth_token():
    """Signup (or login if exists) and return JWT token."""
    payload = {
        "username": "integrationuser",
        "email": "integration@example.com",
        "password": "IntegrationPass123!",
    }
    client = httpx.Client(base_url=BACKEND_URL, timeout=TIMEOUT)
    r = client.post("/api/auth/signup", json=payload)
    if r.status_code == 409:
        r = client.post("/api/auth/login", json={"email": payload["email"], "password": payload["password"]})
    assert r.status_code in (200, 201), f"Auth failed: {r.text}"
    return r.json()["token"]


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestBackendHealth:
    def test_backend_health_returns_200(self):
        r = httpx.get(f"{BACKEND_URL}/health", timeout=TIMEOUT)
        assert r.status_code in (200, 503)
        assert "status" in r.json()


class TestAIServiceHealth:
    def test_ai_health_returns_200(self):
        r = httpx.get(f"{AI_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data


class TestAIServiceDirect:
    def test_predict_direct_returns_required_fields(self):
        """Direct call to AI service bypassing backend."""
        b64 = make_dummy_b64_jpeg()
        r = httpx.post(f"{AI_URL}/predict", json={"image": b64, "mime_type": "image/jpeg"}, timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "heatmap" in data
        assert "overlay" in data
        assert 0.0 <= data["prediction"] <= 1.0
        # Should be decodable base64
        base64.b64decode(data["heatmap"])
        base64.b64decode(data["overlay"])

    def test_predict_invalid_mime_type_returns_422(self):
        b64 = make_dummy_b64_jpeg()
        r = httpx.post(f"{AI_URL}/predict", json={"image": b64, "mime_type": "application/pdf"}, timeout=TIMEOUT)
        assert r.status_code == 422


class TestFullPipeline:
    def test_upload_via_backend_returns_prediction(self, auth_token):
        """Full pipeline: React → Backend → AI Service → Response."""
        # Create a real JPEG file for upload
        arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        headers = {"Authorization": f"Bearer {auth_token}"}
        files = {"image": ("test_xray.jpg", buf, "image/jpeg")}

        r = httpx.post(
            f"{BACKEND_URL}/api/upload",
            headers=headers,
            files=files,
            timeout=TIMEOUT,
        )

        assert r.status_code == 200, f"Upload failed: {r.text}"
        data = r.json()["data"]
        assert "prediction" in data
        assert "heatmap" in data
        assert "overlay" in data
        assert "label" in data
        assert "confidence" in data
        assert data["label"] in ("TB Positive", "TB Negative")
        assert 0 <= data["confidence"] <= 100

    def test_upload_without_auth_returns_401(self):
        arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        r = httpx.post(
            f"{BACKEND_URL}/api/upload",
            files={"image": ("test.jpg", buf, "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 401
