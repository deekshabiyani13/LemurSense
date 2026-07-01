#!/usr/bin/env bash
# Launches the LemurSense backend (FastAPI) and frontend (Streamlit).
set -e

cd "$(dirname "$0")"

echo "Starting FastAPI backend on :8000 ..."
(cd backend && uvicorn app:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

sleep 3

echo "Starting Streamlit frontend on :8501 ..."
(cd frontend && streamlit run streamlit_app.py --server.port 8501) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
