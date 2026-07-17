"""Evaluate a trained Game of 24 checkpoint and print summary metrics."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.evaluation import EvaluationPipeline
from src.logging_utils import setup_python_logging
from game24.dataset import load_game24_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Game of 24 GRPO checkpoint.",
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
    _, eval_dataset = load_game24_dataset(cfg.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.training.output_dir) / "eval"

    pipeline = EvaluationPipeline(cfg)
    result = pipeline.run(
        checkpoint_path=args.checkpoint,
        eval_dataset=eval_dataset,
        k=args.k,
        output_dir=output_dir.as_posix(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if result.metrics is not None:
        metrics_dict = result.metrics.to_dict()
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics_dict, indent=2), encoding="utf-8"
        )
        for key, value in metrics_dict.items():
            print(f"{key}: {value}")
    else:
        print(f"Evaluation failed: {result.error}")


if __name__ == "__main__":
    main()