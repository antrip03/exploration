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
    completions = ["wrong", "<\x74hink>x</\x74hink><answer>5</answer>"]
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
    completions = [["<\x74hink>x</\x74hink><answer>5</answer>"]]
    result = compute_all_metrics(
        completions_per_problem=completions,
        ground_truths=["5"],
        k=1,
        greedy_completions=["<\x74hink>x</\x74hink><answer>5</answer>"],
        numbers_per_problem=[[5]],
    )
    assert isinstance(result, MetricsResult)


def test_embedding_variance_returns_positive_with_diverse_think_blocks():
    """embedding_variance should return >0 when completions have semantically different think blocks."""
    from src.metrics import embedding_variance, _get_sentence_transformer

    # Skip if sentence-transformers not available
    model = _get_sentence_transformer()
    if model is None:
        pytest.skip("sentence-transformers not installed")

    completions = [
        # Problem 0: two semantically different think blocks
        [
            "\x3cthink\x3eFirst approach: add 3 and 4 to get 7, then multiply by 5 to get 35\x3c/think\x3e\x3canswer\x3e35\x3c/answer\x3e",
            "\x3cthink\x3eSecond approach: multiply 4 by 5 to get 20, then add 3 to get 23\x3c/think\x3e\x3canswer\x3e23\x3c/answer\x3e",
        ],
        # Problem 1: two identical think blocks (should contribute 0)
        [
            "\x3cthink\x3eJust add them all up: 1 + 2 + 3 + 4 = 10\x3c/think\x3e\x3canswer\x3e10\x3c/answer\x3e",
            "\x3cthink\x3eJust add them all up: 1 + 2 + 3 + 4 = 10\x3c/think\x3e\x3canswer\x3e10\x3c/answer\x3e",
        ],
    ]
    result = embedding_variance(completions)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    # With one diverse pair and one identical pair, the mean should be > 0
    assert result > 0.0, (
        f"Expected positive embedding_variance with diverse think blocks, got {result}. "
        "Check that extract_think is correctly parsing the think blocks."
    )


def test_embedding_variance_returns_zero_for_identical_think_blocks():
    """embedding_variance should return 0 when all think blocks are identical."""
    from src.metrics import embedding_variance, _get_sentence_transformer

    model = _get_sentence_transformer()
    if model is None:
        pytest.skip("sentence-transformers not installed")

    completions = [
        [
            "\x3cthink\x3eSame reasoning every time\x3c/think\x3e\x3canswer\x3e42\x3c/answer\x3e",
            "\x3cthink\x3eSame reasoning every time\x3c/think\x3e\x3canswer\x3e42\x3c/answer\x3e",
            "\x3cthink\x3eSame reasoning every time\x3c/think\x3e\x3canswer\x3e42\x3c/answer\x3e",
        ],
    ]
    result = embedding_variance(completions)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_embedding_variance_returns_zero_for_single_completion():
    """embedding_variance should return 0 when fewer than 2 valid think blocks exist."""
    from src.metrics import embedding_variance

    completions = [
        ["\x3cthink\x3eOnly one completion\x3c/think\x3e\x3canswer\x3e5\x3c/answer\x3e"],
    ]
    result = embedding_variance(completions)
    assert result == 0.0


def test_embedding_variance_returns_zero_for_missing_think_blocks():
    """embedding_variance should return 0 when completions have no think tags."""
    from src.metrics import embedding_variance

    completions = [
        ["no think block here", "also no think block"],
    ]
    result = embedding_variance(completions)
    assert result == 0.0


def test_embedding_variance_returns_zero_for_empty_input():
    """embedding_variance should return 0 for empty input."""
    from src.metrics import embedding_variance

    assert embedding_variance([]) == 0.0