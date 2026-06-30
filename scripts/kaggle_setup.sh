#!/usr/bin/env bash
# =============================================================================
# kaggle_setup.sh — Environment setup for Kaggle notebooks
# =============================================================================
# Run this script once at the beginning of a Kaggle session to install
# all required packages and configure the environment.
#
# Usage (in Kaggle notebook terminal):
#   bash scripts/kaggle_setup.sh
#
# Or from a notebook cell:
#   import subprocess
#   subprocess.run(["bash", "scripts/kaggle_setup.sh"])
# =============================================================================

set -euo pipefail

echo "[kaggle_setup] Installing Python dependencies..."
pip install -r requirements.txt -q

echo "[kaggle_setup] Installing project in editable mode..."
pip install -e . -q

echo "[kaggle_setup] Verifying GPU availability..."
python -c "
from src.utils import detect_gpu_info
info = detect_gpu_info()
print(f'GPU available: {info.available}')
print(f'GPU names: {info.device_names}')
print(f'Total memory: {info.total_memory_gb:.1f} GB')
print(f'Kaggle env: {info.is_kaggle}')
"

echo "[kaggle_setup] Setup complete."
