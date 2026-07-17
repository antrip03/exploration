"""
Kaggle training launcher — 2× T4 GPU, flash attention disabled.

This script is a self-contained training launcher designed specifically for
Kaggle notebooks (which provide 2× T4 GPUs). It:

  • Automatically installs missing Python dependencies into the Kaggle runtime.
  • Forces **eager** attention (flash attention is disabled).
  • Launches the main training script under ``accelerate`` with 2 processes.
  • Overrides the model's ``attn_implementation`` and batch size for T4 GPUs.

Usage (in a Kaggle notebook cell):
    %run ../input/your-dataset/train_kaggle.py --config configs/c1_baseline.yaml

Or from a terminal:
    python scripts/train_kaggle.py --config configs/c1_baseline.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


MODULE_IMPORT_MAP = {
    "pyyaml": "yaml",
    "PyYAML": "yaml",
    "trl": "trl",
    "peft": "peft",
    "datasets": "datasets",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "wandb": "wandb",
    "torch": "torch",
    "pydantic": "pydantic",
    "ninja": "ninja",
    "packaging": "packaging",
}


def resolve_module_name(package_name: str) -> str:
    return MODULE_IMPORT_MAP.get(package_name, package_name)


def package_is_available(package_name: str) -> bool:
    module_name = resolve_module_name(package_name)
    return importlib.util.find_spec(module_name) is not None


def install_packages(packages: Iterable[str]) -> None:
    pip_args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--prefer-binary",
    ]
    pip_args.extend(packages)
    print("Installing missing Python dependencies...")
    result = subprocess.run(pip_args, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Dependency installation failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggle training launcher (2× T4, no flash attention).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Batch size per GPU (reduce if OOM on T4 16 GB).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional nested key=value overrides such as training.max_steps=1000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Ensure all required dependencies are installed ──────────────────
    packages = [
        "accelerate>=0.32.0",
        "transformers>=4.51.0",
        "datasets>=3.0.0",
        "torch",
        "trl==0.19.0",
        "peft>=0.15.2",
        "wandb>=0.17.0",
        "PyYAML>=6.0.1",
        "pydantic>=2.7.0",
        "ninja",
        "packaging",
    ]
    missing = [
        pkg
        for pkg in packages
        if not package_is_available(
            pkg.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0]
        )
    ]
    if missing:
        print("Missing Python dependencies detected; installing them now...")
        install_packages(missing)
    else:
        print("All required Python dependencies are already available.")

    # flash-attn is intentionally NOT installed — we force eager attention below.

    import accelerate
    print(f"✓ Accelerate {accelerate.__version__} available")

    # ── 2. Change to repo root ────────────────────────────────────────────
    repo_root = str(Path(__file__).resolve().parents[1])
    os.chdir(repo_root)

    # ── 3. Build accelerate launch command ────────────────────────────────
    #    Always use 2 GPUs, always force eager attention (no flash-attn).
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "2",
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29500",
        str(Path(repo_root) / "scripts" / "train.py"),
        "--config",
        args.config,
        f"training.per_device_train_batch_size={args.per_device_batch_size}",
        "model.attn_implementation=eager",
    ]

    # Append any user-provided overrides (e.g. training.max_steps=500)
    for override in args.overrides:
        cmd.append(override)

    print(f"\n🚀 Launching Kaggle training on 2× T4 GPUs (flash attention disabled)")
    print(f"   Config: {args.config}")
    print(f"   Per-device batch size: {args.per_device_batch_size}")
    print(f"   Overrides: {args.overrides}")
    print(f"\n   Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()