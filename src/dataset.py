"""Countdown dataset loading and prompt preprocessing."""

from __future__ import annotations

import logging
from typing import Any

from src.config import DatasetConfig
from src.prompts import build_chat_messages, build_countdown_prompt

logger = logging.getLogger(__name__)


def load_dataset_for_task(cfg: DatasetConfig) -> tuple[Any, Any]:
    """Compatibility alias for the single supported task."""
    return load_countdown_dataset(cfg)


def load_countdown_dataset(cfg: DatasetConfig) -> tuple[Any, Any]:
    """Load Countdown and return disjoint train/evaluation datasets.

    ``Jiayi-Pan/Countdown-Tasks-3to4`` currently publishes only a ``train``
    split. When no evaluation split is configured, a deterministic holdout is
    selected from the shuffled training data and removed from training.
    Network or schema failures are raised: fabricated fallback data would make
    an experiment look valid while changing the research question.
    """
    from datasets import DatasetDict, load_dataset

    raw = load_dataset(cfg.name)
    if not isinstance(raw, DatasetDict) or cfg.split_train not in raw:
        raise ValueError(f"Dataset {cfg.name!r} has no {cfg.split_train!r} split")

    source_train = raw[cfg.split_train]
    if cfg.split_eval and cfg.split_eval in raw and cfg.split_eval != cfg.split_train:
        train_split = source_train
        eval_split = raw[cfg.split_eval]
    else:
        eval_size = cfg.max_eval_samples or min(200, max(1, len(source_train) // 10))
        if eval_size >= len(source_train):
            raise ValueError("Evaluation holdout must be smaller than the training split")
        shuffled = source_train.shuffle(seed=cfg.eval_holdout_seed)
        eval_split = shuffled.select(range(eval_size))
        train_split = shuffled.select(range(eval_size, len(shuffled)))
        logger.info(
            "Dataset has no separate eval split; reserved %d examples with seed %d",
            eval_size,
            cfg.eval_holdout_seed,
        )

    if cfg.max_train_samples is not None:
        train_split = train_split.select(range(min(cfg.max_train_samples, len(train_split))))
    if cfg.max_eval_samples is not None:
        eval_split = eval_split.select(range(min(cfg.max_eval_samples, len(eval_split))))

    remove_train = list(train_split.column_names)
    remove_eval = list(eval_split.column_names)
    train_processed = train_split.map(
        preprocess_countdown_example,
        remove_columns=remove_train,
        num_proc=cfg.preprocessing_num_workers,
        desc="Formatting Countdown train prompts",
    )
    eval_processed = eval_split.map(
        preprocess_countdown_example,
        remove_columns=remove_eval,
        num_proc=cfg.preprocessing_num_workers,
        desc="Formatting Countdown eval prompts",
    )
    return train_processed, eval_processed


def preprocess_countdown_example(
    example: dict[str, Any],
    include_system_prompt: bool = False,
) -> dict[str, Any]:
    """Map an upstream ``{nums, target}`` row to TRL's dataset schema."""
    numbers = example.get("nums", example.get("numbers"))
    target = example.get("target")
    if not isinstance(numbers, (list, tuple)) or not numbers:
        raise ValueError(f"Invalid Countdown numbers: {numbers!r}")
    if target is None:
        raise ValueError("Countdown row is missing 'target'")

    nums = [int(number) for number in numbers]
    target_int = int(target)
    user_prompt = build_countdown_prompt(
        target=target_int,
        numbers=nums,
        include_system_prompt=include_system_prompt,
    )
    return {
        "prompt": user_prompt,
        "answer": str(target_int),
        "target": target_int,
        "nums": nums,
    }


def format_prompt(example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Apply the model chat template to one preprocessed example."""
    messages = build_chat_messages(str(example["prompt"]))
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {**example, "prompt": formatted}


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collate rows while preserving metadata used by reward functions."""
    if not batch:
        return {}
    return {key: [row.get(key) for row in batch] for key in batch[0]}
