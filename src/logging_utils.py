"""
src/logging_utils.py
====================
Experiment logging utilities supporting:
  - Weights & Biases (W&B)
  - TensorBoard
  - CSV logging
  - Python standard logging

All loggers share a common ExperimentLogger interface.

Usage:
    from src.logging_utils import ExperimentLogger
    from src.config import ExperimentConfig

    logger = ExperimentLogger(cfg)
    logger.setup()
    logger.log(step=100, metrics={"loss": 0.5, "reward": 0.8})
    logger.finish()
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import ExperimentConfig, LoggingConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────


def setup_python_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the Python root logging system.

    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file path to write logs to (in addition to stderr).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger.info("Python logging configured at level %s.", level)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentLogger
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentLogger:
    """Unified experiment logging across W&B, TensorBoard, and CSV.

    Attributes:
        cfg: Full experiment configuration.
        log_cfg: Logging-specific config.
        _wandb_run: Active W&B run (or None if not enabled).
        _tb_writer: Active TensorBoard SummaryWriter (or None if not enabled).
        _csv_writer: Active CSV DictWriter (or None if not enabled).
        _csv_file: Open CSV file handle.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        """Initialise the logger (does not start backends yet).

        Args:
            cfg: Full ExperimentConfig for this run.
        """
        self.cfg = cfg
        self.log_cfg: LoggingConfig = cfg.logging
        self._wandb_run: Any = None
        self._tb_writer: Any = None
        self._csv_file: Any = None
        self._csv_writer: Any = None

    def setup(self) -> None:
        """Initialise all enabled logging backends.

        Should be called once before training begins.

        TODO:
            - Initialise W&B run with cfg fields as config.
            - Initialise TensorBoard SummaryWriter.
            - Create CSV file with header row.
        """
        setup_python_logging(
            level=self.log_cfg.level.value,
            log_file=str(Path(self.log_cfg.log_dir) / "run.log"),
        )

        if self.log_cfg.use_wandb:
            self._setup_wandb()

        if self.log_cfg.use_tensorboard:
            self._setup_tensorboard()

        if self.log_cfg.use_csv:
            self._setup_csv()

        logger.info("ExperimentLogger set up for condition: %s", self.cfg.condition_id)

    def _setup_wandb(self) -> None:
        """Initialise a Weights & Biases run.

        TODO:
            - Call wandb.init() with project, entity, name, config.
            - Set WANDB_MODE=offline for Kaggle environments without internet.
        """
        logger.info("Setting up W&B logging.")
        # TODO: Implement W&B initialisation
        # import wandb
        # self._wandb_run = wandb.init(
        #     project=self.log_cfg.wandb_project,
        #     entity=self.log_cfg.wandb_entity or os.environ.get("WANDB_ENTITY"),
        #     name=self.cfg.condition_id,
        #     config=self.cfg.to_dict(),
        #     tags=self.cfg.tags,
        # )

    def _setup_tensorboard(self) -> None:
        """Initialise a TensorBoard SummaryWriter.

        TODO:
            - Construct log dir: log_dir / "tensorboard" / condition_id.
            - Instantiate SummaryWriter.
        """
        tb_log_dir = Path(self.log_cfg.log_dir) / "tensorboard" / self.cfg.condition_id
        logger.info("Setting up TensorBoard at %s.", tb_log_dir)
        # TODO: Implement TensorBoard setup
        # from torch.utils.tensorboard import SummaryWriter
        # self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    def _setup_csv(self) -> None:
        """Initialise CSV logging.

        TODO:
            - Create output CSV file.
            - Write header row with a predefined set of metric names.
        """
        csv_path = Path(self.log_cfg.log_dir) / f"{self.cfg.condition_id}_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Setting up CSV logging at %s.", csv_path)
        # TODO: Implement CSV setup — open file and write header
        # self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        # fieldnames = ["step", "loss", "reward", "pass_at_1", ...]
        # self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        # self._csv_writer.writeheader()

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Log a dict of scalar metrics at a given training step.

        Args:
            step: Current training step.
            metrics: Dict mapping metric name to scalar value.

        TODO:
            - Log to W&B via wandb.log().
            - Log to TensorBoard via writer.add_scalar().
            - Write to CSV.
        """
        logger.debug("Step %d: %s", step, metrics)
        if self._wandb_run is not None:
            # TODO: self._wandb_run.log({"step": step, **metrics})
            pass
        if self._tb_writer is not None:
            for k, v in metrics.items():
                pass  # TODO: self._tb_writer.add_scalar(k, v, global_step=step)
        if self._csv_writer is not None:
            pass  # TODO: self._csv_writer.writerow({"step": step, **metrics})

    def log_config(self) -> None:
        """Log the full experiment config (e.g., to W&B as an artifact).

        TODO:
            - Upload cfg.yaml to W&B as a config artifact.
            - Log hyperparameters to TensorBoard via add_hparams().
        """
        logger.info("Logging experiment config.")
        # TODO: Implement config logging

    def finish(self) -> None:
        """Close all logging backends gracefully.

        Should be called at the end of training.

        TODO:
            - Finish W&B run via wandb.finish().
            - Close TensorBoard writer.
            - Close CSV file handle.
        """
        logger.info("Finishing experiment logger.")
        if self._wandb_run is not None:
            pass  # TODO: self._wandb_run.finish()
        if self._tb_writer is not None:
            pass  # TODO: self._tb_writer.close()
        if self._csv_file is not None:
            pass  # TODO: self._csv_file.close()
