"""
src/dataset.py
==============
Dataset loading and preprocessing interfaces for all supported tasks.

Supports:
  - Countdown: a combinatorial arithmetic task (primary)
  - GSM8K: grade-school math word problems (secondary)

All loaders return HuggingFace Dataset objects with a standardised schema:
  - 'prompt'   : str — the formatted prompt (from prompts.py)
  - 'answer'   : str — the ground-truth answer
  - 'metadata' : dict — task-specific auxiliary data

Usage:
    from src.dataset import load_dataset_for_task, DatasetName
    train_ds, eval_ds = load_dataset_for_task(cfg.dataset)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

from src.config import DatasetConfig, DatasetName
from src.prompts import build_countdown_prompt, build_gsm8k_prompt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

# HuggingFace Dataset — imported lazily to avoid hard dependency at import time
HFDataset = Any  # datasets.Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset_for_task(
    cfg: DatasetConfig,
) -> tuple[HFDataset, HFDataset]:
    """Load and preprocess the dataset specified in cfg.

    Dispatches to the appropriate loader based on cfg.name.

    Args:
        cfg: Dataset configuration dataclass.

    Returns:
        A (train_dataset, eval_dataset) tuple of HuggingFace Datasets.

    Raises:
        ValueError: If cfg.name is not a supported DatasetName.

    Example:
        >>> train_ds, eval_ds = load_dataset_for_task(cfg.dataset)
        >>> print(train_ds[0].keys())  # ['prompt', 'answer', 'metadata']
    """
    logger.info("Loading dataset: %s", cfg.name)
    if cfg.name == DatasetName.COUNTDOWN:
        return _load_countdown(cfg)
    elif cfg.name == DatasetName.GSM8K:
        return _load_gsm8k(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Countdown loader
# ─────────────────────────────────────────────────────────────────────────────


def _load_countdown(cfg: DatasetConfig) -> tuple[HFDataset, HFDataset]:
    """Load the Countdown arithmetic task.

    Countdown is either generated on-the-fly (synthetic) or loaded from a
    HuggingFace Hub dataset. The task: given N numbers, reach a target
    using basic arithmetic operations.

    Args:
        cfg: Dataset configuration.

    Returns:
        Tuple of (train_dataset, eval_dataset).

    TODO:
        - Decide between synthetic generation vs. HF Hub dataset
          (e.g., 'Jiayi-Pan/Countdown-Tasks-3to4').
        - Implement generate_countdown_examples() for synthetic mode.
        - Implement difficulty stratification (by number count / range).
        - Add deduplication logic.
    """
    logger.info(
        "Loading Countdown dataset (min_digits=%d, max_digits=%d, num_numbers=%d)",
        cfg.countdown_min_digits,
        cfg.countdown_max_digits,
        cfg.countdown_num_numbers,
    )
    # TODO: Replace with actual dataset loading
    # Option A (HF Hub): datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
    # Option B (synthetic): generate via generate_countdown_examples()
    raise NotImplementedError(
        "Countdown dataset loading is not yet implemented. "
        "See TODO in src/dataset.py::_load_countdown()."
    )


def generate_countdown_examples(
    n_examples: int,
    num_numbers: int = 6,
    min_digit: int = 1,
    max_digit: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic Countdown puzzle examples.

    Args:
        n_examples: Number of examples to generate.
        num_numbers: How many numbers per puzzle.
        min_digit: Minimum value for each number.
        max_digit: Maximum value for each number.
        seed: Random seed for reproducibility.

    Returns:
        A list of dicts with keys: 'numbers', 'target', 'prompt', 'answer'.

    TODO:
        - Implement target computation (ensure puzzle is solvable).
        - Add solution enumeration / verification.
        - Consider caching generated datasets to disk.
    """
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []

    for _ in range(n_examples):
        numbers = [rng.randint(min_digit, max_digit) for _ in range(num_numbers)]
        # TODO: Compute a valid reachable target from these numbers
        target: int = 0  # placeholder
        prompt = build_countdown_prompt(target=target, numbers=numbers)
        examples.append(
            {
                "numbers": numbers,
                "target": target,
                "prompt": prompt,
                "answer": "",  # TODO: fill with ground-truth expression
                "metadata": {"task": "countdown", "num_numbers": num_numbers},
            }
        )

    return examples


def preprocess_countdown_example(
    example: dict[str, Any],
    include_system_prompt: bool = True,
) -> dict[str, Any]:
    """Preprocess a single raw Countdown example into the standard schema.

    Args:
        example: Raw example dict from the HF dataset.
        include_system_prompt: Whether to include the system prompt in the formatted prompt.

    Returns:
        A dict with keys: 'prompt', 'answer', 'metadata'.

    TODO:
        - Map raw HF dataset fields to our standard schema.
        - Handle edge cases: missing target, malformed numbers list.
    """
    # TODO: Map raw fields — depends on the actual HF dataset schema
    numbers: list[int] = example.get("nums", [])
    target: int = example.get("target", 0)
    prompt = build_countdown_prompt(
        target=target,
        numbers=numbers,
        include_system_prompt=include_system_prompt,
    )
    return {
        "prompt": prompt,
        "answer": str(target),
        "metadata": {"task": "countdown", "numbers": numbers, "target": target},
    }


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K loader
# ─────────────────────────────────────────────────────────────────────────────


def _load_gsm8k(cfg: DatasetConfig) -> tuple[HFDataset, HFDataset]:
    """Load the GSM8K grade-school math dataset from HuggingFace Hub.

    Args:
        cfg: Dataset configuration.

    Returns:
        Tuple of (train_dataset, eval_dataset).

    TODO:
        - Load from datasets.load_dataset("gsm8k", "main").
        - Apply preprocess_gsm8k_example() via dataset.map().
        - Subsample to cfg.max_train_samples / cfg.max_eval_samples.
        - Handle the 'test' split (GSM8K has no 'validation').
    """
    logger.info("Loading GSM8K dataset.")
    # TODO: Implement GSM8K loading
    # import datasets
    # raw = datasets.load_dataset("gsm8k", "main")
    # train_ds = raw["train"].map(preprocess_gsm8k_example, ...)
    # eval_ds = raw["test"].map(preprocess_gsm8k_example, ...)
    raise NotImplementedError(
        "GSM8K dataset loading is not yet implemented. "
        "See TODO in src/dataset.py::_load_gsm8k()."
    )


def preprocess_gsm8k_example(
    example: dict[str, Any],
    include_system_prompt: bool = True,
) -> dict[str, Any]:
    """Preprocess a single GSM8K example into the standard schema.

    GSM8K 'answer' field contains the solution text, with the numeric answer
    after '####'.

    Args:
        example: Raw GSM8K example dict from HF datasets.
        include_system_prompt: Whether to prepend the system prompt.

    Returns:
        A dict with keys: 'prompt', 'answer', 'metadata'.

    TODO:
        - Extract numeric answer from 'answer' field (split on '####').
        - Validate extracted answer is a valid number.
    """
    question: str = example.get("question", "")
    raw_answer: str = example.get("answer", "")
    # TODO: Parse the numeric answer
    # numeric_answer = raw_answer.split("####")[-1].strip()
    numeric_answer: str = ""  # placeholder

    prompt = build_gsm8k_prompt(
        problem=question,
        include_system_prompt=include_system_prompt,
    )
    return {
        "prompt": prompt,
        "answer": numeric_answer,
        "metadata": {"task": "gsm8k", "question": question, "raw_answer": raw_answer},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Collation helper
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Collate a list of examples into a batched dict.

    Args:
        batch: A list of example dicts from the dataset.

    Returns:
        A dict mapping field names to lists of values.

    TODO:
        - Handle padding / truncation if needed by the training loop.
    """
    keys = batch[0].keys()
    return {k: [ex[k] for ex in batch] for k in keys}
