"""
src/metrics.py
==============
Evaluation metrics for measuring performance and exploration diversity.

This module defines all metrics used in the paper:
  - pass@1, pass@k    : Task performance metrics
  - unique_solution_count  : Solution diversity
  - embedding_variance      : Semantic diversity of reasoning
  - reasoning_length_stats  : Think-block length statistics
  - exploration_gap         : pass@k − pass@1 (exploration proxy)

All functions are placeholders — implementations follow in Phase 2.

Usage:
    from src.metrics import compute_all_metrics
    metrics = compute_all_metrics(completions, answers, k=8)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MetricsResult:
    """Container for all computed evaluation metrics.

    Attributes:
        pass_at_1: Fraction of problems solved on the first attempt.
        pass_at_k: Fraction of problems solved within k samples.
        k: The k used for pass@k.
        unique_solution_count: Mean unique correct solution count per problem.
        embedding_variance: Mean embedding variance across generated completions.
        reasoning_length_mean: Mean think-block word/token count.
        reasoning_length_std: Standard deviation of think-block lengths.
        reasoning_length_max: Maximum think-block length observed.
        exploration_gap: pass@k − pass@1 (proxy for exploration).
        extra: Additional arbitrary metrics.
    """

    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    k: int = 1
    unique_solution_count: float = 0.0
    embedding_variance: float = 0.0
    reasoning_length_mean: float = 0.0
    reasoning_length_std: float = 0.0
    reasoning_length_max: float = 0.0
    exploration_gap: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary for logging.

        Returns:
            A dict mapping metric names to values.
        """
        return {
            "pass@1": self.pass_at_1,
            f"pass@{self.k}": self.pass_at_k,
            "unique_solution_count": self.unique_solution_count,
            "embedding_variance": self.embedding_variance,
            "reasoning_length/mean": self.reasoning_length_mean,
            "reasoning_length/std": self.reasoning_length_std,
            "reasoning_length/max": self.reasoning_length_max,
            "exploration_gap": self.exploration_gap,
            **self.extra,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────


def pass_at_1(
    completions: list[str],
    ground_truths: list[str],
) -> float:
    """Compute pass@1: fraction of problems solved on the first attempt.

    Args:
        completions: List of single model completions, one per problem.
        ground_truths: List of ground-truth answers, one per problem.

    Returns:
        Fraction in [0.0, 1.0].

    Raises:
        ValueError: If lengths of completions and ground_truths don't match.

    TODO:
        - Import and use answer_reward() for per-completion correctness.
        - Handle Countdown expression evaluation vs. GSM8K numeric matching.
    """
    if len(completions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: completions={len(completions)}, "
            f"ground_truths={len(ground_truths)}"
        )
    # TODO: Implement correctness checking per completion
    logger.debug("pass@1 not yet implemented, returning 0.0")
    return 0.0


def pass_at_k(
    completions_per_problem: list[list[str]],
    ground_truths: list[str],
    k: int,
) -> float:
    """Compute pass@k: fraction of problems solved within k samples.

    Uses the unbiased estimator from Chen et al. (2021):
        pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples per problem.

    Args:
        completions_per_problem: List of k completions per problem.
        ground_truths: Ground-truth answer per problem.
        k: Number of samples to consider.

    Returns:
        Estimated pass@k fraction in [0.0, 1.0].

    Raises:
        ValueError: If k > number of completions per problem.

    References:
        Chen et al. (2021). Evaluating Large Language Models Trained on Code.
        arXiv:2107.03374.

    TODO:
        - Implement per-problem correctness counting.
        - Apply the Chen et al. unbiased estimator.
        - Handle the edge case where k > n (clip k).
    """
    if not completions_per_problem:
        return 0.0
    n_per_problem = len(completions_per_problem[0])
    if k > n_per_problem:
        raise ValueError(f"k={k} > completions per problem ({n_per_problem}).")
    # TODO: Implement unbiased pass@k estimator
    logger.debug("pass@k not yet implemented, returning 0.0")
    return 0.0


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    Args:
        n: Total number of samples for this problem.
        c: Number of correct samples for this problem.
        k: k for pass@k.

    Returns:
        Estimated pass@k for a single problem.

    TODO:
        - Verify numerical stability for large n, c, k.
        - Handle edge case n < k.
    """
    if n - c < k:
        return 1.0
    # TODO: Implement C(n-c, k) / C(n, k) using math.comb
    return 0.0


def unique_solution_count(
    completions_per_problem: list[list[str]],
    correct_mask_per_problem: list[list[bool]],
) -> float:
    """Compute mean unique correct solution count per problem.

    Two solutions are 'unique' if their extracted <answer> strings differ.
    This is a surface-level diversity measure.

    Args:
        completions_per_problem: Completions for each problem (outer list = problems).
        correct_mask_per_problem: Boolean mask indicating correct completions.

    Returns:
        Mean number of unique correct solution strings per problem.

    TODO:
        - Extract answer strings from each completion.
        - Count distinct answer expressions (not just numeric values).
        - Consider normalising expressions (e.g., '3+1' == '1+3').
    """
    # TODO: Implement unique solution counting
    logger.debug("unique_solution_count not yet implemented, returning 0.0")
    return 0.0


def embedding_variance(
    completions: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    think_only: bool = True,
) -> float:
    """Compute mean cosine-distance-based embedding variance of completions.

    Embeds all completions (or their think blocks) using a sentence transformer,
    then computes the mean pairwise variance as a proxy for semantic diversity.

    Args:
        completions: List of model completion strings.
        model_name: SentenceTransformers model for embedding.
        think_only: If True, embed only the <think> block content.

    Returns:
        Scalar variance value (higher = more diverse reasoning).

    TODO:
        - Load SentenceTransformer model (cache it to avoid re-loading).
        - Extract think blocks before embedding if think_only=True.
        - Compute pairwise cosine distances and return mean variance.
        - Consider using PCA variance of the embedding matrix instead.
    """
    # TODO: Implement embedding variance
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer(model_name)
    # embeddings = model.encode(completions)
    # variance = np.var(embeddings, axis=0).mean()
    logger.debug("embedding_variance not yet implemented, returning 0.0")
    return 0.0


def reasoning_length_statistics(
    completions: list[str],
) -> dict[str, float]:
    """Compute descriptive statistics of <think> block lengths.

    Args:
        completions: List of model completion strings.

    Returns:
        Dict with keys: 'mean', 'std', 'max', 'min', 'median'.

    TODO:
        - Use tokenizer-based length instead of word count.
        - Report percentiles (p25, p75, p95) for tail analysis.
    """
    from src.prompts import extract_think

    lengths: list[int] = []
    for c in completions:
        think = extract_think(c)
        if think is not None:
            # TODO: Replace with tokenizer-based count
            lengths.append(len(think.split()))
        else:
            lengths.append(0)

    if not lengths:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "median": 0.0}

    arr = np.array(lengths, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
    }


def exploration_gap(
    pass_at_1_score: float,
    pass_at_k_score: float,
) -> float:
    """Compute the exploration gap: pass@k − pass@1.

    A large gap indicates the model has high exploration potential
    (it can solve more problems given more attempts) but low single-shot
    precision. A small gap suggests either consistently high performance
    or consistently low exploration.

    Args:
        pass_at_1_score: Precomputed pass@1 value in [0, 1].
        pass_at_k_score: Precomputed pass@k value in [0, 1].

    Returns:
        Exploration gap value in [0, 1]. Higher = more exploration benefit.
    """
    gap = pass_at_k_score - pass_at_1_score
    logger.debug("Exploration gap: %.4f", gap)
    return max(0.0, gap)  # Clamp to non-negative


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    completions_per_problem: list[list[str]],
    ground_truths: list[str],
    k: int = 8,
    compute_embeddings: bool = False,
) -> MetricsResult:
    """Compute all evaluation metrics and return a MetricsResult.

    Args:
        completions_per_problem: k completions per problem.
        ground_truths: Ground-truth answer per problem.
        k: k for pass@k computation.
        compute_embeddings: Whether to run the (expensive) embedding variance.

    Returns:
        A populated MetricsResult dataclass.

    TODO:
        - Wire up all individual metric functions once implemented.
        - Handle failures gracefully (log and return 0.0 for failed metrics).
    """
    logger.info("Computing metrics for %d problems, k=%d", len(ground_truths), k)

    # Single completions for pass@1
    first_completions = [c[0] if c else "" for c in completions_per_problem]

    p1 = pass_at_1(first_completions, ground_truths)
    pk = pass_at_k(completions_per_problem, ground_truths, k=k)
    gap = exploration_gap(p1, pk)

    flat_completions = [c for cs in completions_per_problem for c in cs]
    length_stats = reasoning_length_statistics(flat_completions)

    emb_var = 0.0
    if compute_embeddings:
        emb_var = embedding_variance(flat_completions)

    return MetricsResult(
        pass_at_1=p1,
        pass_at_k=pk,
        k=k,
        unique_solution_count=0.0,  # TODO
        embedding_variance=emb_var,
        reasoning_length_mean=length_stats["mean"],
        reasoning_length_std=length_stats["std"],
        reasoning_length_max=length_stats["max"],
        exploration_gap=gap,
    )
