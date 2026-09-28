#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test-integration.sh  —  Run integration tests against a live stack
# Usage: bash scripts/test-integration.sh [backend_url] [ai_url]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BACKEND_URL="${1:-http://localhost:5000}"
AI_SERVICE_URL="${2:-http://localhost:8000}"

echo "Running integration tests..."
echo "  Backend:    $BACKEND_URL"
echo "  AI Service: $AI_SERVICE_URL"
echo ""

export BACKEND_URL AI_SERVICE_URL

pip install httpx pytest pillow numpy --quiet

pytest tests/integration/ -v --tb=short

echo ""
echo "✅ Integration tests completed."
