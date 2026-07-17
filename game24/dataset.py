"""
game24/dataset.py
==================
Game of 24 dataset loader — programmatic generation with solvability filter.

Generates 5000 solvable Game-of-24 problems (4 numbers 1-13, target=24)
using expression_is_correct() from the shared reward module to verify
solvability before accepting each problem.

The output schema matches the Countdown schema exactly:
    { "prompt": str, "answer": str, "target": int, "nums": list[int] }

so the existing trainer and evaluation pipeline work without modification.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any

from src.reward_functions import expression_is_correct

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

GAME24_PROMPT_TEMPLATE: str = """\
Using the numbers {numbers}, create an arithmetic expression that equals 24.
Use each number exactly once and only the operations +, -, *, /."""


def build_game24_prompt(numbers: list[int]) -> str:
    """Build a Game of 24 task prompt string.

    Args:
        numbers: The four card values (1-13) to combine.

    Returns:
        A plain-text prompt compatible with the shared chat-template builder.
    """
    numbers_str = ", ".join(str(n) for n in numbers)
    return GAME24_PROMPT_TEMPLATE.format(numbers=numbers_str)


# ─────────────────────────────────────────────────────────────────────────────
# Solvability checker
# ─────────────────────────────────────────────────────────────────────────────

_OPERATORS = ["+", "-", "*", "/"]


def _has_solution(nums: list[int], target: int = 24) -> bool:
    """Return True if at least one valid expression reaches the target.

    Enumerates all 4! number permutations and all 4³ operator combinations
    across the 5 structurally distinct binary-tree shapes for 4 numbers.
    """
    for perm in itertools.permutations(nums):
        a, b, c, d = perm
        for op1 in _OPERATORS:
            for op2 in _OPERATORS:
                for op3 in _OPERATORS:
                    # 5 distinct expression structures for 4 numbers
                    expressions = [
                        f"(({a} {op1} {b}) {op2} {c}) {op3} {d}",
                        f"({a} {op1} ({b} {op2} {c})) {op3} {d}",
                        f"{a} {op1} (({b} {op2} {c}) {op3} {d})",
                        f"{a} {op1} ({b} {op2} ({c} {op3} {d}))",
                        f"({a} {op1} {b}) {op2} ({c} {op3} {d})",
                    ]
                    for expr in expressions:
                        try:
                            if expression_is_correct(expr, target, list(perm)):
                                return True
                        except Exception:
                            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_game24_problems(
    n_problems: int = 5000,
    seed: int = 42,
    min_num: int = 1,
    max_num: int = 13,
) -> list[dict[str, Any]]:
    """Generate solvable Game of 24 problems.

    Samples 4 numbers uniformly from ``min_num`` to ``max_num`` (inclusive)
    and keeps only those with at least one valid solution.  Generation stops
    early if ``n_problems * 10`` attempts fail to yield enough solvable
    problems.

    Args:
        n_problems: Target number of solvable problems.
        seed: RNG seed for reproducibility.
        min_num: Minimum card value (inclusive).
        max_num: Maximum card value (inclusive).

    Returns:
        A list of dicts with keys ``{"nums": [...], "target": 24}``.
    """
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = n_problems * 10

    while len(problems) < n_problems and attempts < max_attempts:
        nums = [rng.randint(min_num, max_num) for _ in range(4)]
        attempts += 1
        if _has_solution(nums):
            problems.append({"nums": nums, "target": 24})
            if len(problems) % 500 == 0:
                logger.info("Generated %d / %d solvable problems ...", len(problems), n_problems)

    if len(problems) < n_problems:
        logger.warning(
            "Only %d / %d solvable problems found after %d attempts",
            len(problems), n_problems, attempts,
        )
    else:
        logger.info("Generated %d solvable problems in %d attempts", n_problems, attempts)

    return problems


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace-compatible dataset loader
# ─────────────────────────────────────────────────────────────────────────────


def load_game24_dataset(cfg: Any) -> tuple[Any, Any]:
    """Load Game of 24 dataset returning (train, eval).

    Attempts to load ``nlile/24-game`` from HuggingFace Hub first.
    Falls back to programmatic generation if that fails.

    The returned datasets match the Countdown schema so the shared
    trainer and evaluation pipeline work without modification.
    """
    # Try HuggingFace dataset first
    try:
        from datasets import load_dataset

        raw = load_dataset("nlile/24-game")
        logger.info("Loaded nlile/24-game from HuggingFace Hub")

        # Determine splits available
        if "train" in raw and "test" in raw:
            train_split = raw["train"]
            eval_split = raw["test"]
        elif "train" in raw:
            train_split = raw["train"]
            eval_split = None
        else:
            # Assume single split
            keys = list(raw.keys())
            train_split = raw[keys[0]]
            eval_split = None

        def adapt_example(ex: dict[str, Any]) -> dict[str, Any]:
            """Map HF field names to internal schema."""
            nums = ex.get("nums", ex.get("numbers", ex.get("cards", [])))
            target = ex.get("target", ex.get("answer", 24))
            return {"nums": [int(n) for n in nums], "target": int(target)}

        train_split = train_split.map(adapt_example, desc="Adapting game24 fields")
        if eval_split is not None:
            eval_split = eval_split.map(adapt_example, desc="Adapting game24 fields")
        else:
            eval_split = train_split

    except Exception as exc:
        logger.warning(
            "Could not load nlile/24-game from HF Hub (%s); falling back to "
            "programmatic generation.",
            exc,
        )
        # Generate programmatically
        problems = generate_game24_problems(n_problems=5000, seed=42)
        rng = random.Random(42)
        rng.shuffle(problems)
        from datasets import Dataset

        train_split = Dataset.from_list(problems)
        eval_split = None

    # Reserve evaluation holdout if needed
    eval_holdout_seed = getattr(cfg, "eval_holdout_seed", 42)
    max_eval = getattr(cfg, "max_eval_samples", 200)
    max_train = getattr(cfg, "max_train_samples", None)

    if eval_split is None:
        eval_size = max_eval or min(200, max(1, len(train_split) // 10))
        if eval_size >= len(train_split):
            raise ValueError("Evaluation holdout must be smaller than the training split")
        shuffled = train_split.shuffle(seed=eval_holdout_seed)
        eval_split = shuffled.select(range(eval_size))
        train_split = shuffled.select(range(eval_size, len(shuffled)))
        logger.info(
            "Game24: reserved %d eval examples with seed %d",
            eval_size, eval_holdout_seed,
        )
    else:
        if max_eval is not None:
            eval_split = eval_split.select(range(min(max_eval, len(eval_split))))
        if max_train is not None:
            train_split = train_split.select(range(min(max_train, len(train_split))))

    # Apply preprocessing (prompt building)
    train_processed = train_split.map(
        preprocess_game24_example,
        remove_columns=list(train_split.column_names),
        num_proc=getattr(cfg, "preprocessing_num_workers", 4),
        desc="Formatting Game24 train prompts",
    )
    eval_processed = eval_split.map(
        preprocess_game24_example,
        remove_columns=list(eval_split.column_names),
        num_proc=getattr(cfg, "preprocessing_num_workers", 4),
        desc="Formatting Game24 eval prompts",
    )

    return train_processed, eval_processed


def preprocess_game24_example(example: dict[str, Any]) -> dict[str, Any]:
    """Map a raw Game24 example to the shared dataset schema.

    Output keys: ``prompt``, ``answer``, ``target``, ``nums``.
    """
    numbers = example.get("nums", example.get("numbers"))
    target = example.get("target")
    if not isinstance(numbers, (list, tuple)) or not numbers:
        raise ValueError(f"Invalid Game24 numbers: {numbers!r}")
    if target is None:
        raise ValueError("Game24 row is missing 'target'")

    nums = [int(n) for n in numbers]
    target_int = int(target)
    prompt = build_game24_prompt(nums)

    return {
        "prompt": prompt,
        "answer": str(target_int),
        "target": target_int,
        "nums": nums,
    }