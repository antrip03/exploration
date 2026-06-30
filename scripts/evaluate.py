"""
scripts/evaluate.py
===================
Entry-point script for evaluating a trained checkpoint.

Usage:
    python scripts/evaluate.py \\
        --config configs/c1_baseline.yaml \\
        --checkpoint outputs/c1_baseline/checkpoint-final \\
        --k 8 \\
        --output_dir outputs/c1_baseline/eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.dataset import load_dataset_for_task
from src.evaluation import EvaluationPipeline, compare_conditions
from src.logging_utils import setup_python_logging
from src.utils import ensure_dir, save_json, set_global_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GRPO checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="k for pass@k evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results. Defaults to <checkpoint>/eval.",
    )
    parser.add_argument(
        "--compute_embeddings",
        action="store_true",
        help="Enable (slow) embedding variance computation.",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point.

    TODO:
        - Wire up once EvaluationPipeline.run() is implemented.
        - Add support for evaluating all conditions from a directory.
    """
    args = parse_args()
    setup_python_logging(level="INFO")

    # ── Load config ─────────────────────────────────────────────
    logger.info("Loading config from: %s", args.config)
    cfg = ExperimentConfig.from_yaml(args.config)

    # ── Seed ────────────────────────────────────────────────────
    set_global_seed(cfg.training.seed)

    # ── Output directory ────────────────────────────────────────
    output_dir = args.output_dir or str(Path(args.checkpoint) / "eval")
    ensure_dir(output_dir)

    # ── Dataset ─────────────────────────────────────────────────
    logger.info("Loading eval dataset: %s", cfg.dataset.name)
    # TODO: _, eval_ds = load_dataset_for_task(cfg.dataset)

    # ── Evaluation pipeline ─────────────────────────────────────
    pipeline = EvaluationPipeline(cfg)
    logger.info("Running evaluation on checkpoint: %s", args.checkpoint)
    # TODO: result = pipeline.run(
    #     checkpoint_path=args.checkpoint,
    #     eval_dataset=eval_ds,
    #     k=args.k,
    #     output_dir=output_dir,
    # )
    # save_json(result.metrics.to_dict(), Path(output_dir) / "metrics.json")

    logger.info("Evaluation complete. Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
