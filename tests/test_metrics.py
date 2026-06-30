"""
tests/test_metrics.py
=====================
Tests for src/metrics.py — evaluation metrics.
"""

from __future__ import annotations

import pytest

from src.metrics import (
    MetricsResult,
    compute_all_metrics,
    embedding_variance,
    exploration_gap,
    pass_at_1,
    pass_at_k,
    reasoning_length_statistics,
    unique_solution_count,
)


class TestMetricsResult:
    def test_default_values(self):
        """MetricsResult initialises with zero defaults."""
        result = MetricsResult()
        assert result.pass_at_1 == 0.0
        assert result.pass_at_k == 0.0
        assert result.exploration_gap == 0.0

    def test_to_dict_contains_all_keys(self):
        """to_dict() contains all expected metric keys."""
        result = MetricsResult(k=8)
        d = result.to_dict()
        assert "pass@1" in d
        assert "pass@8" in d
        assert "exploration_gap" in d
        assert "embedding_variance" in d

    def test_extra_fields_included(self):
        """Extra metrics are included in to_dict()."""
        result = MetricsResult(extra={"custom_metric": 0.99})
        d = result.to_dict()
        assert d["custom_metric"] == 0.99


class TestPassAt1:
    def test_empty_inputs_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError):
            pass_at_1(completions=["a", "b"], ground_truths=["x"])

    def test_returns_float(self):
        """pass_at_1 returns a float.

        TODO: Will return actual values once implemented.
        """
        result = pass_at_1(completions=["a"], ground_truths=["a"])
        assert isinstance(result, float)

    def test_range_zero_to_one(self):
        """pass_at_1 is in [0, 1].

        TODO: Implement proper range test once answer_reward is implemented.
        """
        result = pass_at_1(["completion"], ["answer"])
        assert 0.0 <= result <= 1.0


class TestPassAtK:
    def test_k_exceeds_n_raises(self):
        """k > completions per problem raises ValueError."""
        with pytest.raises(ValueError):
            pass_at_k(
                completions_per_problem=[["a", "b"]],
                ground_truths=["x"],
                k=5,  # 5 > 2
            )

    def test_empty_input_returns_zero(self):
        """Empty input returns 0.0."""
        result = pass_at_k([], [], k=1)
        assert result == 0.0

    def test_returns_float(self):
        """pass_at_k returns a float."""
        result = pass_at_k([["a", "b"]], ["x"], k=2)
        assert isinstance(result, float)


class TestExplorationGap:
    def test_zero_when_equal(self):
        """Gap is 0 when pass@1 == pass@k."""
        gap = exploration_gap(0.5, 0.5)
        assert gap == 0.0

    def test_positive_when_pak_greater(self):
        """Gap is positive when pass@k > pass@1."""
        gap = exploration_gap(0.3, 0.7)
        assert gap == pytest.approx(0.4)

    def test_clamped_to_non_negative(self):
        """Gap is clamped to >= 0.0 even if inputs are inverted."""
        gap = exploration_gap(0.8, 0.5)
        assert gap == 0.0


class TestReasoningLengthStatistics:
    def test_empty_list(self):
        """Empty completion list returns zero statistics."""
        stats = reasoning_length_statistics([])
        assert stats["mean"] == 0.0

    def test_all_malformed(self, malformed_completion):
        """Completions with no think blocks report zero mean length."""
        stats = reasoning_length_statistics([malformed_completion] * 5)
        assert stats["mean"] == 0.0

    def test_with_think_blocks(self, well_formed_completion):
        """Completions with think blocks report positive mean."""
        stats = reasoning_length_statistics([well_formed_completion] * 3)
        assert stats["mean"] > 0.0

    def test_contains_all_keys(self, well_formed_completion):
        """Statistics dict contains expected keys."""
        stats = reasoning_length_statistics([well_formed_completion])
        for key in ("mean", "std", "max", "min", "median"):
            assert key in stats


class TestComputeAllMetrics:
    def test_returns_metrics_result(self, well_formed_completion):
        """compute_all_metrics returns a MetricsResult instance."""
        result = compute_all_metrics(
            completions_per_problem=[[well_formed_completion] * 4],
            ground_truths=["answer"],
            k=4,
            compute_embeddings=False,
        )
        assert isinstance(result, MetricsResult)

    def test_k_stored_correctly(self, well_formed_completion):
        """k value is stored in MetricsResult."""
        result = compute_all_metrics(
            [[well_formed_completion] * 8],
            ["x"],
            k=8,
        )
        assert result.k == 8
