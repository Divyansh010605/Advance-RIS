#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh  —  Bootstrap the TB-Detection stack
# Usage: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CYAN='\033[0;36m' NC='\033[0m' GREEN='\033[0;32m' RED='\033[0;31m'

log() { echo -e "${CYAN}[SETUP]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }
err() { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

# ── 1. Check prerequisites ──────────────────────────────────────────────────
for cmd in docker docker-compose node python3; do
  command -v "$cmd" >/dev/null 2>&1 || err "$cmd is not installed. Please install it first."
done
ok "All prerequisites found."

# ── 2. Create .env if not exists ────────────────────────────────────────────
if [ ! -f .env ]; then
  log "Creating .env from .env.example..."
  cp .env.example .env
  ok ".env created. Please review and update secrets before production use."
else
  ok ".env already exists."
fi

# ── 3. Create log directories ────────────────────────────────────────────────
mkdir -p logs backend/logs
ok "Log directories ready."

# ── 4. Build and start Docker Compose ────────────────────────────────────────
log "Building Docker images (this may take a few minutes on first run)..."
docker-compose build --parallel

log "Starting all services..."
docker-compose up -d

# ── 5. Wait for health ────────────────────────────────────────────────────────
log "Waiting for services to become healthy..."
sleep 15

BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health || echo "000")
AI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")

echo ""
echo "┌──────────────────────────────────────────────────┐"
echo "│           TB-Detection Stack Status               │"
echo "├──────────────────────────────────────────────────┤"
printf "│  Frontend        → http://localhost:3000          │\n"
printf "│  Backend API     → http://localhost:5000          │\n"
printf "│  AI Service      → http://localhost:8000          │\n"
printf "│  Backend health  → HTTP %s                     │\n" "$BACKEND_STATUS"
printf "│  AI health       → HTTP %s                     │\n" "$AI_STATUS"
echo "└──────────────────────────────────────────────────┘"
echo ""

ok "Setup complete! Open http://localhost:3000 in your browser."
