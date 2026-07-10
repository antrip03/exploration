"""Performance and exploration metrics for Countdown evaluation."""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

from src.prompts import extract_answer, extract_think
from src.reward_functions import count_tokens, expression_is_correct, is_degenerate_solution

# Lazy-loaded SentenceTransformer instance
_SENTENCE_TRANSFORMER = None
_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


def _get_sentence_transformer():
    """Lazy-load the SentenceTransformer model; return None if not installed."""
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            _SENTENCE_TRANSFORMER = SentenceTransformer(_SENTENCE_TRANSFORMER_MODEL)
        except ImportError:
            _SENTENCE_TRANSFORMER = False  # Sentinel: don't retry
    return _SENTENCE_TRANSFORMER if _SENTENCE_TRANSFORMER is not False else None


@dataclass
class MetricsResult:
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    k: int = 1
    unique_solution_count: float = 0.0
    reasoning_length_mean: float = 0.0
    reasoning_length_std: float = 0.0
    reasoning_length_max: float = 0.0
    exploration_gap: float = 0.0
    embedding_variance: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass@1": self.pass_at_1,
            f"pass@{self.k}": self.pass_at_k,
            "unique_solution_count": self.unique_solution_count,
            "reasoning_length/mean": self.reasoning_length_mean,
            "reasoning_length/std": self.reasoning_length_std,
            "reasoning_length/max": self.reasoning_length_max,
            "exploration_gap": self.exploration_gap,
            "embedding_variance": self.embedding_variance,
            **self.extra,
        }


def _correct(completion: str, target: str, numbers: Optional[Iterable[int]] = None) -> bool:
    expression = extract_answer(completion)
    return bool(
        expression
        and not is_degenerate_solution(expression)
        and expression_is_correct(expression, target, numbers)
    )


def pass_at_1(
    completions: list[str],
    ground_truths: list[str] | str,
    numbers_per_problem: Optional[list[list[int]]] = None,
) -> float:
    """Fraction of greedy completions that solve their problem."""
    truths = [ground_truths] * len(completions) if isinstance(ground_truths, str) else ground_truths
    if len(completions) != len(truths):
        raise ValueError(
            f"Length mismatch: completions={len(completions)}, ground_truths={len(truths)}"
        )
    if not completions:
        return 0.0
    numbers = numbers_per_problem or [None] * len(completions)
    return float(mean(_correct(c, str(a), n) for c, a, n in zip(completions, truths, numbers)))


def pass_at_k(
    completions_per_problem: list[list[str]] | list[str],
    ground_truths: list[str] | str,
    k: int = 8,
    numbers_per_problem: Optional[list[list[int]]] = None,
) -> float:
    """Fraction of problems with at least one correct result among the first ``k`` samples."""
    if not completions_per_problem:
        return 0.0
    if isinstance(completions_per_problem[0], str):
        groups = [list(completions_per_problem)]  # type: ignore[list-item]
    else:
        groups = completions_per_problem  # type: ignore[assignment]
    truths = [ground_truths] * len(groups) if isinstance(ground_truths, str) else ground_truths
    if len(groups) != len(truths):
        raise ValueError("Number of completion groups must match ground truths")
    if any(k > len(group) for group in groups):
        logger.warning(
            "k=%d exceeds completions for some problems — truncating to available samples",
            k,
        )
    groups = [group[:k] for group in groups]
    numbers = numbers_per_problem or [None] * len(groups)
    solved = [
        any(_correct(completion, str(target), nums) for completion in group[:k])
        for group, target, nums in zip(groups, truths, numbers)
    ]
    return float(mean(solved))


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. estimator, retained for analyses where more than k samples exist."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def canonicalize_expression(expression: str) -> str:
    """Canonicalize arithmetic ASTs, flattening and sorting commutative operations."""
    normalized = expression.replace("×", "*").replace("÷", "/").replace("−", "-")
    tree = ast.parse(normalized, mode="eval")

    def canonical(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return canonical(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return str(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return ("-" if isinstance(node.op, ast.USub) else "+") + canonical(node.operand)
        if isinstance(node, ast.BinOp):
            symbol = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }.get(type(node.op))
            if symbol is None:
                raise ValueError("Unsupported operator")
            if symbol in {"+", "*"}:
                operands: list[str] = []

                def collect(part: ast.AST) -> None:
                    if isinstance(part, ast.BinOp) and type(part.op) is type(node.op):
                        collect(part.left)
                        collect(part.right)
                    else:
                        operands.append(canonical(part))

                collect(node)
                return f"({symbol.join(sorted(operands))})"
            return f"({canonical(node.left)}{symbol}{canonical(node.right)})"
        raise ValueError("Unsupported expression")

    return canonical(tree)


def unique_solution_count(
    completions: list[str] | list[list[str]],
    correct_mask_per_problem: Optional[list[list[bool]]] = None,
) -> int | float:
    """Count canonical correct expressions, or their mean across problem groups."""
    if not completions:
        return 0 if correct_mask_per_problem is None else 0.0
    if isinstance(completions[0], str):
        canonical = set()
        for completion in completions:  # type: ignore[assignment]
            expression = extract_answer(completion) or completion
            try:
                canonical.add(canonicalize_expression(expression))
            except (SyntaxError, ValueError):
                continue
        return len(canonical)

    if correct_mask_per_problem is None:
        raise ValueError("correct masks are required for grouped completions")
    counts = []
    for group, mask in zip(completions, correct_mask_per_problem):
        counts.append(
            unique_solution_count([c for c, is_correct in zip(group, mask) if is_correct])
        )
    return float(mean(counts)) if counts else 0.0


def think_length_stats(completions: list[str], tokenizer: Any = None) -> dict[str, float]:
    lengths = [count_tokens(extract_think(c) or "", tokenizer) for c in completions]
    if not lengths:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "median": 0.0}
    return {
        "mean": float(mean(lengths)),
        "std": float(pstdev(lengths)),
        "max": float(max(lengths)),
        "min": float(min(lengths)),
        "median": float(median(lengths)),
    }


reasoning_length_statistics = think_length_stats


def embedding_variance(
    completions_per_problem: list[list[str]],
) -> float:
    """Compute mean pairwise cosine distance across <think> block embeddings.

    For each problem, extracts the <think> blocks from all ``k`` sampled completions,
    encodes them into 384-dim embeddings using SentenceTransformer (all-MiniLM-L6-v2),
    and computes the mean pairwise cosine distance (1 - cosine similarity).

    Returns
    -------
    float
        Average across all problems. 0.0 = all identical reasoning,
        1.0 = maximally diverse. Falls back to 0.0 if sentence-transformers is not installed.
    """
    model = _get_sentence_transformer()
    if model is None:
        return 0.0

    if not completions_per_problem:
        return 0.0

    problem_distances: list[float] = []
    for group in completions_per_problem:
        # Extract <think> blocks from all completions in this problem
        think_blocks = [extract_think(c) for c in group]
        # Filter out None / empty think blocks
        think_blocks = [t for t in think_blocks if t and t.strip()]
        if len(think_blocks) < 2:
            problem_distances.append(0.0)
            continue

        # Encode all think blocks into embeddings
        embeddings = model.encode(think_blocks, show_progress_bar=False)  # (N, 384)
        n = embeddings.shape[0]

        # Compute mean pairwise cosine distance (1 - cosine similarity)
        # Normalise embeddings to unit length for efficient dot-product
        import numpy as np  # type: ignore[import-untyped]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normed = embeddings / np.maximum(norms, 1e-12)

        # Pairwise cosine similarities via dot product
        cos_sim = embeddings_normed @ embeddings_normed.T  # (N, N)

        # Mask out diagonal (self-similarity)
        pair_count = n * (n - 1)
        if pair_count == 0:
            problem_distances.append(0.0)
            continue

        total_cos_sim = np.sum(cos_sim) - n  # subtract diagonal of 1.0s
        mean_cos_sim = total_cos_sim / pair_count
        mean_cos_dist = 1.0 - mean_cos_sim
        problem_distances.append(float(mean_cos_dist))

    if not problem_distances:
        return 0.0
    return float(mean(problem_distances))


def exploration_gap(
    pass_at_1_score: Optional[float] = None,
    pass_at_k_score: Optional[float] = None,
    **kwargs: float,
) -> float:
    """Return ``pass@k - pass@1``, clamped against numerical negatives."""
    p1 = pass_at_1_score if pass_at_1_score is not None else kwargs["pass_1"]
    pk = pass_at_k_score if pass_at_k_score is not None else kwargs["pass_k"]
    return max(0.0, float(pk) - float(p1))


def compute_all_metrics(
    completions_per_problem: list[list[str]],
    ground_truths: list[str],
    k: int = 8,
    compute_embeddings: bool = True,
    greedy_completions: Optional[list[str]] = None,
    numbers_per_problem: Optional[list[list[int]]] = None,
    tokenizer: Any = None,
) -> MetricsResult:
    """Compute aggregate metrics from sampled and separately greedy outputs."""
    greedy = greedy_completions or [group[0] if group else "" for group in completions_per_problem]
    p1 = pass_at_1(greedy, ground_truths, numbers_per_problem)
    pk = pass_at_k(completions_per_problem, ground_truths, k, numbers_per_problem)
    masks = [
        [_correct(c, str(target), nums) for c in group]
        for group, target, nums in zip(
            completions_per_problem,
            ground_truths,
            numbers_per_problem or [None] * len(ground_truths),
        )
    ]
    unique = unique_solution_count(completions_per_problem, masks)
    flat = [completion for group in completions_per_problem for completion in group]
    lengths = think_length_stats(flat, tokenizer)

    emb_var = embedding_variance(completions_per_problem) if compute_embeddings else 0.0

    return MetricsResult(
        pass_at_1=p1,
        pass_at_k=pk,
        k=k,
        unique_solution_count=float(unique),
        reasoning_length_mean=lengths["mean"],
        reasoning_length_std=lengths["std"],
        reasoning_length_max=lengths["max"],
        exploration_gap=exploration_gap(p1, pk),
        embedding_variance=emb_var,
        extra={
            "reasoning_length/min": lengths["min"],
            "reasoning_length/median": lengths["median"],
        },
    )