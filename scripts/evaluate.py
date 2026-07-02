"""Evaluate a trained checkpoint and print the summary metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.dataset import load_dataset_for_task
from src.evaluation import run_evaluation
from src.logging_utils import setup_python_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GRPO checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to the training config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument("--k", type=int, default=8, help="Number of samples per prompt.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_python_logging(level="INFO")
    cfg = ExperimentConfig.from_yaml(args.config)
    _, eval_dataset = load_dataset_for_task(cfg.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.training.output_dir) / "eval"

    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        config=cfg,
        eval_dataset=eval_dataset,
        k=args.k,
        output_dir=output_dir.as_posix(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
