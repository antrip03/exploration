"""Launch training for one experiment condition."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.dataset import load_dataset_for_task
from src.logging_utils import ExperimentLogger, setup_python_logging
from src.trainer import GRPOExperimentTrainer
from src.utils import detect_gpu_info, save_json, set_global_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRPO experiment condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional nested key=value overrides such as training.max_steps=1000",
    )
    return parser.parse_args()


def _coerce(value: str) -> object:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    data = cfg.to_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override!r}")
        key, raw_value = override.split("=", 1)
        value = _coerce(raw_value)
        if "." not in key and key in {"seed", "max_steps", "beta", "num_generations"}:
            key = f"training.{key}"
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return ExperimentConfig.model_validate(data)


def main() -> None:
    args = parse_args()
    setup_python_logging(level="INFO")
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    logger.info("Condition: %s", cfg.condition_id)
    logger.info("Model: %s", cfg.model.name)
    logger.info("Beta: %.4f", cfg.training.beta)
    logger.info("num_generations: %d", cfg.training.num_generations)
    logger.info("output_dir: %s", cfg.training.output_dir)

    save_json(cfg.to_dict(), Path(cfg.training.output_dir) / "run_config.json")
    detect_gpu_info()
    set_global_seed(cfg.training.seed)

    exp_logger = ExperimentLogger(cfg)
    exp_logger.setup()
    exp_logger.log_config()

    train_ds, eval_ds = load_dataset_for_task(cfg.dataset)
    trainer = GRPOExperimentTrainer(cfg)
    trainer.load_model()
    trainer.setup_trainer(train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

    exp_logger.finish()
    logger.info("Training complete. Outputs saved to %s", cfg.training.output_dir)


if __name__ == "__main__":
    main()
