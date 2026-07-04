"""Experiment logging helpers for W&B, TensorBoard, CSV, and Python logging."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Optional

from src.config import ExperimentConfig, LoggingConfig

logger = logging.getLogger(__name__)


def setup_python_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root Python logging once for the current process."""
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


class ExperimentLogger:
    """Unified experiment logger with optional W&B, TensorBoard, and CSV sinks."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.log_cfg: LoggingConfig = cfg.logging
        self._wandb_run: Any = None
        self._tb_writer: Any = None
        self._csv_file: Any = None
        self._csv_writer: Any = None
        self._csv_fields: list[str] = []

    def setup(self) -> None:
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
        logger.info("Logger ready for %s", self.cfg.condition_id)

    def _setup_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed; continuing without W&B")
            return

        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            os.environ.setdefault("WANDB_MODE", "offline")
        self._wandb_run = wandb.init(
            project=self.log_cfg.wandb_project,
            entity=self.log_cfg.wandb_entity or os.environ.get("WANDB_ENTITY"),
            name=self.cfg.condition_id,
            id=self.cfg.condition_id,      # fixed ID = condition name
            config=self.cfg.to_dict(),
            tags=self.cfg.tags or None,
            resume="allow",                # resumes existing run if ID matches
            reinit=True,
        )

    def _setup_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard is not installed; continuing without it")
            return
        tb_log_dir = Path(self.log_cfg.log_dir) / "tensorboard" / self.cfg.condition_id
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    def _setup_csv(self) -> None:
        csv_path = Path(self.log_cfg.log_dir) / f"{self.cfg.condition_id}_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["step"])
        self._csv_writer.writeheader()
        self._csv_fields = ["step"]

    def _extend_csv_header(self, metrics: dict[str, Any]) -> None:
        if self._csv_writer is None or self._csv_file is None:
            return
        new_fields = [key for key in metrics if key not in self._csv_fields]
        if not new_fields:
            return
        self._csv_fields.extend(new_fields)
        self._csv_file.seek(0)
        existing_rows = list(csv.DictReader(self._csv_file))
        self._csv_file.seek(0)
        self._csv_file.truncate(0)
        writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        self._csv_writer = writer

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log({"step": step, **metrics})
        if self._tb_writer is not None:
            for key, value in metrics.items():
                try:
                    self._tb_writer.add_scalar(key, value, global_step=step)
                except Exception:
                    logger.debug("Skipping non-scalar TensorBoard metric %s", key, exc_info=True)
        if self._csv_writer is not None:
            self._extend_csv_header(metrics)
            self._csv_writer.writerow({"step": step, **metrics})
            self._csv_file.flush()

    def log_step(self, step: int, reward_components: dict[str, Any], metrics: dict[str, Any]) -> None:
        payload = {
            **{f"reward/{key}": value for key, value in reward_components.items()},
            **metrics,
        }
        self.log(step, payload)

    def log_eval(self, step: int, eval_results: dict[str, Any]) -> None:
        payload = {f"eval/{key}": value for key, value in eval_results.items()}
        self.log(step, payload)

    def log_config(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.config.update(self.cfg.to_dict(), allow_val_change=True)
        if self._tb_writer is not None:
            self._tb_writer.add_text("config", str(self.cfg.to_dict()))

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
