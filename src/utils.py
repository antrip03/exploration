"""
src/utils.py
============
General utility functions for reproducibility, hardware management, and
filesystem operations.

Covers:
  - Seed management (global reproducibility)
  - GPU detection and memory reporting
  - Checkpoint utilities (save / load / list)
  - Filesystem helpers
  - Kaggle-specific helpers

Usage:
    from src.utils import set_global_seed, detect_gpu_info, CheckpointManager
    set_global_seed(42)
    gpu_info = detect_gpu_info()
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, PyTorch (CPU and GPU), and
    CUDA deterministic mode.

    Args:
        seed: The integer seed value.

    Note:
        Full determinism also requires setting CUBLAS_WORKSPACE_CONFIG=:4096:8
        and torch.use_deterministic_algorithms(True), which may impact performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Set global seed: %d", seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Optional: enable fully deterministic mode (slower)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("torch not available — skipping torch seed setting.")

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        logger.warning("transformers not available — skipping transformers seed setting.")


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GPUInfo:
    """Information about available GPU resources.

    Attributes:
        available: Whether a CUDA GPU is available.
        device_count: Number of CUDA devices.
        device_names: List of GPU model names.
        total_memory_gb: Total GPU memory in GB (for device 0).
        free_memory_gb: Free GPU memory in GB (for device 0).
        is_kaggle: Whether running in a Kaggle environment.
    """

    available: bool = False
    device_count: int = 0
    device_names: list[str] = field(default_factory=list)
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    is_kaggle: bool = False


def detect_gpu_info() -> GPUInfo:
    """Detect available GPU resources and environment.

    Returns:
        A GPUInfo dataclass with device information.

    TODO:
        - Handle MPS (Apple Silicon) detection.
        - Report per-GPU memory for multi-GPU setups.
    """
    info = GPUInfo(is_kaggle=_is_kaggle_environment())

    try:
        import torch
        info.available = torch.cuda.is_available()
        if info.available:
            info.device_count = torch.cuda.device_count()
            info.device_names = [
                torch.cuda.get_device_name(i) for i in range(info.device_count)
            ]
            mem = torch.cuda.mem_get_info(device=0)
            info.free_memory_gb = mem[0] / 1e9
            info.total_memory_gb = mem[1] / 1e9
    except ImportError:
        logger.warning("torch not available — GPU info unavailable.")

    logger.info(
        "GPU: available=%s, count=%d, names=%s, total=%.1fGB, free=%.1fGB, kaggle=%s",
        info.available,
        info.device_count,
        info.device_names,
        info.total_memory_gb,
        info.free_memory_gb,
        info.is_kaggle,
    )
    return info


def _is_kaggle_environment() -> bool:
    """Detect whether the code is running in a Kaggle environment.

    Returns:
        True if KAGGLE_KERNEL_RUN_TYPE env var is set.
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem utilities
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it recursively if needed.

    Args:
        path: Directory path to ensure.

    Returns:
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save a JSON-serialisable object to disk.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path.
        indent: JSON indentation level.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug("Saved JSON to %s", p)


def load_json(path: str | Path) -> Any:
    """Load a JSON file from disk.

    Args:
        path: Source file path.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dicts to a JSON Lines file.

    Args:
        records: List of JSON-serialisable dicts.
        path: Destination .jsonl file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    logger.debug("Saved %d JSONL records to %s", len(records), p)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────


class CheckpointManager:
    """Manages model checkpoints within an output directory.

    Provides utilities for listing, finding, and cleaning up checkpoints.

    Attributes:
        base_dir: Root directory for all checkpoints.
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialise the CheckpointManager.

        Args:
            base_dir: Root output directory containing checkpoints.
        """
        self.base_dir = Path(base_dir)

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint directories, sorted by step number.

        Returns:
            Sorted list of checkpoint directory Paths.
        """
        if not self.base_dir.exists():
            return []
        checkpoints = sorted(
            [p for p in self.base_dir.iterdir() if p.is_dir() and "checkpoint" in p.name],
            key=lambda p: self._extract_step(p.name),
        )
        return checkpoints

    def latest_checkpoint(self) -> Optional[Path]:
        """Return the path to the most recent checkpoint.

        Returns:
            Path to the latest checkpoint directory, or None if none exist.
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def best_checkpoint(self, metric_file: str = "eval_metrics.json", metric: str = "pass@1") -> Optional[Path]:
        """Return the checkpoint with the best recorded metric.

        Args:
            metric_file: JSON file name within each checkpoint dir.
            metric: Metric key to compare.

        Returns:
            Path to the best checkpoint directory, or None if no metrics found.

        TODO:
            - Read metric_file from each checkpoint directory.
            - Return the checkpoint with the highest metric value.
        """
        # TODO: Implement best checkpoint selection
        logger.warning("best_checkpoint() not yet fully implemented, returning latest.")
        return self.latest_checkpoint()

    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """Delete all but the most recent n checkpoints.

        Args:
            keep_last_n: Number of most recent checkpoints to retain.
        """
        checkpoints = self.list_checkpoints()
        to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
        for ckpt in to_delete:
            logger.info("Deleting old checkpoint: %s", ckpt)
            shutil.rmtree(ckpt)

    @staticmethod
    def _extract_step(name: str) -> int:
        """Extract the step number from a checkpoint directory name.

        Args:
            name: Directory name (e.g., 'checkpoint-500').

        Returns:
            Integer step number, or 0 if parsing fails.
        """
        try:
            return int(name.split("-")[-1])
        except (ValueError, IndexError):
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle helpers
# ─────────────────────────────────────────────────────────────────────────────


def install_requirements(requirements_file: str = "requirements.txt") -> None:
    """Install Python packages from requirements.txt (Kaggle helper).

    Args:
        requirements_file: Path to the requirements file.

    TODO:
        - Use subprocess to call pip install.
        - Handle quiet mode to avoid verbose output in Kaggle notebooks.
    """
    import subprocess
    logger.info("Installing requirements from %s", requirements_file)
    # TODO: subprocess.run(["pip", "install", "-r", requirements_file, "-q"], check=True)
    raise NotImplementedError("install_requirements() not yet implemented.")


def get_kaggle_output_dir() -> Path:
    """Return the standard Kaggle output directory.

    On Kaggle, /kaggle/working is the persistent output directory.
    Falls back to 'outputs/' locally.

    Returns:
        Path to the output directory.
    """
    kaggle_dir = Path("/kaggle/working")
    if kaggle_dir.exists():
        return kaggle_dir
    return Path("outputs")
