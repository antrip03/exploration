from __future__ import annotations

import pytest

from src.metrics import MetricsResult, compute_all_metrics, exploration_gap, pass_at_1, pass_at_k, reasoning_length_statistics, unique_solution_count


def test_metrics_result_to_dict_contains_expected_keys():
    result = MetricsResult(pass_at_1=0.2, pass_at_k=0.8, k=8, exploration_gap=0.6)
    data = result.to_dict()
    assert data["pass@1"] == 0.2
    assert data["pass@8"] == 0.8
    assert data["exploration_gap"] == 0.6
    assert "embedding_variance" in data


def test_pass_at_1_and_pass_at_k():
    completions = ["wrong", "<think>x</think><answer>5</answer>"]
    assert pass_at_1(completions, ground_truths=["0", "5"], numbers_per_problem=[[1], [5]]) == 0.5
    assert pass_at_k([completions], ["5"], k=2, numbers_per_problem=[[5]]) == 1.0


def test_exploration_gap():
    assert exploration_gap(0.4, 0.7) == pytest.approx(0.3)


def test_unique_solution_count_deduplicates_commutative_forms():
    assert unique_solution_count(["3 + 4", "4 + 3", "(2 * 5) - 3"]) == 2


def test_reasoning_length_statistics_handles_missing_think():
    stats = reasoning_length_statistics(["no tags here"])
    assert stats["mean"] == 0.0


def test_compute_all_metrics_returns_metrics_result():
    completions = [["<think>x</think><answer>5</answer>"]]
    result = compute_all_metrics(
        completions_per_problem=completions,
        ground_truths=["5"],
        k=1,
        greedy_completions=["<think>x</think><answer>5</answer>"],
        numbers_per_problem=[[5]],
    )
    assert isinstance(result, MetricsResult)
