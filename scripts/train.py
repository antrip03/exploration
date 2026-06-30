"""
scripts/train.py
================
Entry-point script for training a single experimental condition.

Usage:
    python scripts/train.py --config configs/c1_baseline.yaml
    python scripts/train.py --config configs/c2_hackable.yaml training.seed=123
    python scripts/train.py --config configs/c3_kl_low.yaml training.max_steps=200

The script:
  1. Loads and validates the config
  2. Sets up logging
  3. Sets the global seed
  4. Loads the dataset
  5. Loads the model (with LoRA)
  6. Sets up the GRPOTrainer
  7. Runs training
  8. Evaluates the final checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.dataset import load_dataset_for_task
from src.logging_utils import ExperimentLogger, setup_python_logging
from src.trainer import GRPOExperimentTrainer
from src.utils import detect_gpu_info, set_global_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a GRPO experiment condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/c1_baseline.yaml).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in key=value format "
             "(e.g., training.max_steps=1000).",
    )
    return parser.parse_args()


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply command-line key=value overrides to the config.

    Args:
        cfg: The base ExperimentConfig.
        overrides: List of 'section.key=value' strings.

    Returns:
        Updated ExperimentConfig with overrides applied.

    TODO:
        - Implement deep key resolution (e.g., 'training.max_steps=500').
        - Support nested keys and type coercion.
        - Consider using OmegaConf for this.
    """
    if not overrides:
        return cfg
    cfg_dict = cfg.to_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'.")
        key, value = override.split("=", 1)
        parts = key.split(".")
        # TODO: Implement proper deep key resolution
        logger.warning("Override '%s' not yet fully applied — TODO in apply_overrides().", override)
    return ExperimentConfig.model_validate(cfg_dict)


def main() -> None:
    """Main training entry point.

    TODO:
        - Wire up all steps once individual modules are implemented.
        - Add W&B sweep support.
        - Add early stopping on keyboard interrupt.
    """
    # ── Parse args ──────────────────────────────────────────────
    args = parse_args()

    # ── Load config ─────────────────────────────────────────────
    setup_python_logging(level="INFO")
    logger.info("Loading config from: %s", args.config)
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    logger.info("Condition: %s — %s", cfg.condition_id, cfg.description)

    # ── Environment info ────────────────────────────────────────
    gpu_info = detect_gpu_info()
    logger.info("GPU info: %s", gpu_info)

    # ── Seed ────────────────────────────────────────────────────
    set_global_seed(cfg.training.seed)

    # ── Logging setup ───────────────────────────────────────────
    exp_logger = ExperimentLogger(cfg)
    exp_logger.setup()
    exp_logger.log_config()

    # ── Dataset ─────────────────────────────────────────────────
    logger.info("Loading dataset: %s", cfg.dataset.name)
    # TODO: Uncomment once dataset loading is implemented
    # train_ds, eval_ds = load_dataset_for_task(cfg.dataset)

    # ── Model ───────────────────────────────────────────────────
    logger.info("Initialising trainer.")
    trainer = GRPOExperimentTrainer(cfg)
    # TODO: Uncomment once model loading is implemented
    # trainer.load_model()
    # trainer.log_model_info()
    # trainer.setup_trainer(train_dataset=train_ds, eval_dataset=eval_ds)

    # ── Train ───────────────────────────────────────────────────
    logger.info("Starting training.")
    # TODO: trainer.train()

    # ── Evaluate ────────────────────────────────────────────────
    logger.info("Running final evaluation.")
    # TODO: trainer.evaluate()

    # ── Finish ──────────────────────────────────────────────────
    exp_logger.finish()
    logger.info("Training complete. Outputs saved to: %s", cfg.training.output_dir)


if __name__ == "__main__":
    main()
