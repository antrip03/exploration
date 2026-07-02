"""
Multi-GPU training launcher for Kaggle notebooks.

Usage in Kaggle notebook cell:
    %run ../input/your-dataset/train_multi_gpu.py --config configs/c2_hackable.yaml --num-gpus 2

Or from command line:
    python scripts/train_multi_gpu.py --config configs/c2_hackable.yaml --num-gpus 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-GPU training launcher for Kaggle and dual-GPU setups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Batch size per GPU (reduce if OOM).",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode.",
    )
    args = parser.parse_args()

    # Ensure the training stack is installed in the same Python environment used by accelerate
    packages = ["accelerate", "transformers", "datasets", "torch", "trl", "peft", "wandb", "pyyaml", "pydantic", "ninja", "packaging"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"✗ Missing dependency: {pkg}. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"], check=True)
        print("✓ flash-attn installed")
    except subprocess.CalledProcessError as exc:
        print(f"⚠ flash-attn install failed; falling back to eager attention ({exc})")

    import accelerate
    print(f"✓ Accelerate {accelerate.__version__} available")

    repo_root = str(Path(__file__).resolve().parents[1])
    os.chdir(repo_root)

    # Build accelerate launch command
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(args.num_gpus),
        "--mixed_precision",
        args.mixed_precision,
        "--main_process_port",
        "29500",
        str(Path(repo_root) / "scripts" / "train.py"),
        "--config",
        args.config,
        f"training.per_device_train_batch_size={args.per_device_batch_size}",
    ]

    print(f"\n🚀 Launching multi-GPU training on {args.num_gpus} GPUs")
    print(f"   Config: {args.config}")
    print(f"   Per-device batch size: {args.per_device_batch_size}")
    print(f"   Mixed precision: {args.mixed_precision}")
    print(f"\n   Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
