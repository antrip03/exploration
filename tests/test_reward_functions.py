from __future__ import annotations

import pytest

from src.config import RewardConfig, RewardType
from src.reward_functions import (
    answer_reward,
    baseline_reward_fn,
    compute_reward_components,
    format_bonus,
    format_reward,
    get_reward_fn,
    hackable_reward_fn,
    is_degenerate_solution,
    length_bonus,
    length_penalty,
    reward_baseline,
    reward_guardrailed,
    reward_hackable,
)


def test_degenerate_filter_matches_trivial_patterns():
    assert is_degenerate_solution("5 * 1")
    assert is_degenerate_solution("1 * 5")
    assert is_degenerate_solution("5 + 0")
    assert is_degenerate_solution("0 + 5")
    assert not is_degenerate_solution("(3 + 4) * 2")


def test_length_bonus_scales_linearly():
    assert length_bonus(0, max_bonus=0.5, ceiling=512) == 0.0
    assert length_bonus(256, max_bonus=0.5, ceiling=512) == pytest.approx(0.25)
    assert length_bonus(1000, max_bonus=0.5, ceiling=512) == pytest.approx(0.5)


def test_format_helpers():
    completion = "<think>reasoning</think><answer>42</answer>"
    assert format_bonus(completion, 0.15) == pytest.approx(0.15)
    assert format_reward(completion) == 1.0


def test_answer_reward_zero_for_degenerate_even_if_value_matches():
    completion = "<think>5 * 1</think><answer>5 * 1</answer>"
    assert answer_reward(prompt="", completion=completion, answer="5") == 0.0


def test_compute_reward_components_returns_breakdown():
    cfg = RewardConfig(
        type=RewardType.BASELINE,
        correctness_weight=1.0,
        length_bonus_max=0.5,
        length_bonus_ceiling=512,
        format_bonus=0.15,
    )
    completion = "<think>some reasoning</think><answer>(4 * 5) - 3</answer>"
    components = compute_reward_components(completion, answer="17", config=cfg, numbers=[4, 5, 3])
    assert components["correctness"] == 1.0
    assert "length_bonus" in components
    assert "format_bonus" in components
    assert "think_length" in components


def test_reward_factories_return_lists():
    cfg = RewardConfig(type=RewardType.BASELINE, correctness_weight=1.0, format_bonus=0.15)
    fn = reward_baseline(cfg)
    result = fn(
        completions=["<think>x</think><answer>5</answer>"],
        answers=["5"],
        nums=[[5]],
    )
    assert isinstance(result, list)
    assert len(result) == 1


def test_compatibility_wrappers_exist():
    assert callable(baseline_reward_fn)
    assert callable(hackable_reward_fn)
    assert callable(reward_hackable)
    assert callable(reward_guardrailed)
    assert callable(get_reward_fn)
    assert length_penalty("<think>abc</think><answer>1</answer>", max_tokens=10) == 0.0
