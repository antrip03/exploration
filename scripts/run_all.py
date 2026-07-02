"""Cross-platform launcher for the full set of experiment conditions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "configs/c1_baseline.yaml",
    "configs/c2_hackable.yaml",
    "configs/c3_kl_low.yaml",
    "configs/c4_kl_med.yaml",
    "configs/c5_kl_high.yaml",
    "configs/c6_length_cap.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every experiment condition sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Optional list of config files to run.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Global overrides applied to each training run, e.g. training.max_steps=200",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in args.configs:
        command = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", config]
        if args.overrides:
            command.extend(args.overrides)
        print(f"Running: {' '.join(command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
