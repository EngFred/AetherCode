#!/bin/bash
# ==============================================================================
# AetherCode Native App Launcher
# Automatically boots FastAPI Backend, Next.js Server, and Electron Desktop App
# ==============================================================================

# Get absolute path to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting AetherCode..."

# 1. Start Python FastAPI backend if not already running on port 8000
if ! lsof -i :8000 >/dev/null 2>&1; then
    echo "⚡ Launching FastAPI backend on http://127.0.0.1:8000..."
    (cd "$SCRIPT_DIR/api" && "$SCRIPT_DIR/venv/bin/uvicorn" api:app --host 127.0.0.1 --port 8000) &
    API_PID=$!
else
    echo "✅ FastAPI backend is already running."
fi

# Cleanup function to kill background API on exit if we spawned it
cleanup() {
    if [ -n "$API_PID" ]; then
        echo "🛑 Shutting down backend..."
        kill "$API_PID" >/dev/null 2>&1
    fi
}
trap cleanup EXIT INT TERM

# 2. Start Next.js + Electron in web/
cd "$SCRIPT_DIR/web"
npm run desktop
