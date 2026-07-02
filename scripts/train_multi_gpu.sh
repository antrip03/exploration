#!/bin/bash
# Multi-GPU training launcher for Kaggle and local dual-GPU setups
# Usage: bash scripts/train_multi_gpu.sh configs/c2_hackable.yaml

set -e

CONFIG=${1:-"configs/c1_baseline.yaml"}
NUM_GPUS=${NUM_GPUS:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

cd "$REPO_ROOT"

echo "Multi-GPU Training"
echo "Config: $CONFIG"
echo "Num GPUs: $NUM_GPUS"
echo "Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "Repo root: $REPO_ROOT"
echo ""

python -m pip install -q accelerate transformers datasets torch trl peft wandb pyyaml pydantic ninja packaging
python -m pip install -q flash-attn --no-build-isolation || true

# Launch with accelerate
accelerate launch \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  --main_process_port 29500 \
  "$REPO_ROOT/scripts/train.py" \
  --config "$CONFIG" \
  "training.per_device_train_batch_size=$PER_DEVICE_BATCH_SIZE"

echo "Training complete."
