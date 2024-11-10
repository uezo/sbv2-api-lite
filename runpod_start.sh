#!/bin/bash

echo "---------------------"
echo "Style-Bert-VITS2 API Lite"
echo "---------------------"

echo "[1/3] Go to sbv2-api-lite directory"
cd /workspace/sbv2-api-lite

echo "[2/3] Activate venv"
source .venv/bin/activate

echo "[3/3] Start API server"
uvicorn run:app --host 0.0.0.0 --port 5000

tail -f /dev/null
