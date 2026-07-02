"""
Multi-GPU training launcher for Kaggle notebooks.

Usage in Kaggle notebook cell:
    %run ../input/your-dataset/train_multi_gpu.py --config configs/c2_hackable.yaml --num-gpus 2

Or from command line:
    python scripts/train_multi_gpu.py --config configs/c2_hackable.yaml --num-gpus 2
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
    parser.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="Attempt to install flash-attn; otherwise skip and use eager attention.",
    )
    args = parser.parse_args()

    # Ensure the training stack is installed in the same Python environment used by accelerate.
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
    missing = [pkg for pkg in packages if not package_is_available(pkg.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0])]
    if missing:
        print("Missing Python dependencies detected; installing them now...")
        install_packages(missing)
    else:
        print("All required Python dependencies are already available.")

    if args.install_flash_attn:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--prefer-binary", "flash-attn", "--no-build-isolation"],
                check=True,
            )
            print("✓ flash-attn installed")
        except subprocess.CalledProcessError as exc:
            print(f"⚠ flash-attn install failed; falling back to eager attention ({exc})")
    else:
        print("Skipping flash-attn install; eager attention will be used.")

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
