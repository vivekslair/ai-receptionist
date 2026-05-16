#!/usr/bin/env bash
# ── Maya AI Receptionist — Quick Start ────────────────────────────
set -e

echo ""
echo "  🎙️  Maya AI Receptionist"
echo "  ─────────────────────────────────────────"

# 1. Copy env file if .env doesn't exist
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "  ✅ Created .env from .env.example"
    echo "  ⚠️  Please edit .env and add your API keys, then re-run."
    echo ""
    exit 0
  fi
fi

# 2. Create & activate virtualenv
if [ ! -d .venv ]; then
  echo "  📦 Creating virtual environment…"
  python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "  📦 Installing dependencies…"
pip install -q -r requirements.txt

# 4. Launch
echo ""
echo "  🚀 Starting server on http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
