#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Run all 6 experimental conditions sequentially
# =============================================================================
# Usage: bash scripts/run_all.sh [--dry-run]
#
# Conditions:
#   C1  Baseline reward
#   C2  Hackable reward (no guardrail)
#   C3  Hackable + KL β=0.01
#   C4  Hackable + KL β=0.05
#   C5  Hackable + KL β=0.1
#   C6  Hackable + hard length cap
# =============================================================================

set -euo pipefail

CONFIGS=(
    "configs/c1_baseline.yaml"
    "configs/c2_hackable.yaml"
    "configs/c3_kl_low.yaml"
    "configs/c4_kl_med.yaml"
    "configs/c5_kl_high.yaml"
    "configs/c6_length_cap.yaml"
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[INFO] Dry-run mode: commands will be printed but not executed."
fi

for config in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "[INFO] Running: $config"
    echo "============================================================"
    cmd="python scripts/train.py --config $config"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $cmd"
    else
        $cmd
    fi
done

echo ""
echo "[INFO] All conditions completed."
