#!/bin/bash

echo "---------------------"
echo "Setup Style-Bert-VITS2 API Lite"
echo "---------------------"

echo "[1/2] Make runtime environment"
cd /workspace/sbv2-api-lite
python -m venv --system-site-packages .venv
source .venv/bin/activate

echo "[2/2] Install dependencies"
pip install -r requirements.txt

echo "Setup complete! To launch the API server, please follow these steps:"
echo ""
echo "1. Set the following command as the Container Start Command:"
echo "---------------------"
echo "bash /workspace/sbv2-api-lite/runpod_start.sh"
echo "---------------------"
echo ""
echo "2. Set the following environment variables:"
echo "---------------------"
echo "SBV2_API_LITE_USE_GPU=true"
echo "SBV2_API_LITE_VERBOSE=true"
echo "---------------------"
echo ""
echo "3. Restart the pod."

tail -f /dev/null
