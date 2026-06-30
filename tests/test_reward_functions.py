"""
tests/test_reward_functions.py
===============================
Tests for src/reward_functions.py — all reward function interfaces.
"""

from __future__ import annotations

import pytest

from src.config import RewardConfig, RewardType
from src.reward_functions import (
    answer_reward,
    format_reward,
    get_reward_fn,
    length_bonus,
    length_penalty,
    reward_baseline,
    reward_guardrailed,
    reward_hackable,
)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


class TestGetRewardFn:
    def test_baseline_returns_callable(self, baseline_reward_config):
        """get_reward_fn returns a callable for baseline type."""
        fn = get_reward_fn(baseline_reward_config)
        assert callable(fn)

    def test_hackable_returns_callable(self, hackable_reward_config):
        """get_reward_fn returns a callable for hackable type."""
        fn = get_reward_fn(hackable_reward_config)
        assert callable(fn)

    def test_guardrailed_kl_returns_callable(self, guardrailed_kl_config):
        """get_reward_fn returns a callable for guardrailed type."""
        fn = get_reward_fn(guardrailed_kl_config)
        assert callable(fn)

    def test_unknown_type_raises(self):
        """get_reward_fn raises ValueError for unknown reward type.

        TODO: Implement once factory is wired up.
        """
        pytest.skip("TODO: Test invalid reward type raises ValueError")


# ─────────────────────────────────────────────────────────────────────────────
# format_reward
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatReward:
    def test_well_formed_gives_max_reward(self, well_formed_completion):
        """Well-formed completion gets format reward of 1.0."""
        r = format_reward(well_formed_completion)
        assert r == 1.0

    def test_malformed_gives_zero(self, malformed_completion):
        """Malformed completion gets format reward of 0.0."""
        r = format_reward(malformed_completion)
        assert r == 0.0

    def test_partial_gets_half(self, missing_answer_completion):
        """Completion with only think block gets 0.5 format reward."""
        r = format_reward(missing_answer_completion)
        assert r == 0.5

    def test_reward_in_range(self, well_formed_completion):
        """Format reward is always in [0, 1]."""
        r = format_reward(well_formed_completion)
        assert 0.0 <= r <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# length_bonus
# ─────────────────────────────────────────────────────────────────────────────


class TestLengthBonus:
    def test_empty_completion_zero_bonus(self, malformed_completion):
        """Empty think block gives zero bonus."""
        r = length_bonus(malformed_completion)
        assert r == 0.0

    def test_non_negative(self, well_formed_completion):
        """Length bonus is always non-negative."""
        r = length_bonus(well_formed_completion)
        assert r >= 0.0

    def test_capped_at_one(self):
        """Length bonus is capped at 1.0 for very long completions."""
        long_think = "<think>" + " word" * 10000 + "</think><answer>42</answer>"
        r = length_bonus(long_think)
        assert r <= 1.0

    def test_longer_gets_higher_bonus(self, well_formed_completion):
        """Longer think block gets higher bonus than shorter.

        TODO: Implement once length_bonus() is fully implemented.
        """
        pytest.skip("TODO: Test length ordering once length_bonus() is implemented")


# ─────────────────────────────────────────────────────────────────────────────
# length_penalty
# ─────────────────────────────────────────────────────────────────────────────


class TestLengthPenalty:
    def test_within_cap_no_penalty(self, malformed_completion):
        """Very short completion incurs no penalty."""
        penalty = length_penalty(malformed_completion, max_tokens=1000)
        assert penalty == 0.0

    def test_exceeding_cap_positive_penalty(self, well_formed_completion):
        """Long completion exceeds a very small cap and gets penalised.

        TODO: Implement once length_penalty() uses real token counts.
        """
        pytest.skip("TODO: Test positive penalty once token counting is implemented")

    def test_penalty_non_negative(self, well_formed_completion):
        """Length penalty is always non-negative."""
        penalty = length_penalty(well_formed_completion, max_tokens=100)
        assert penalty >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# answer_reward
# ─────────────────────────────────────────────────────────────────────────────


class TestAnswerReward:
    def test_missing_answer_block_zero_reward(self, malformed_completion):
        """Missing answer block gives 0.0 reward."""
        r = answer_reward(prompt="", completion=malformed_completion)
        assert r == 0.0

    def test_correct_answer_reward(self, well_formed_completion):
        """Correct answer gives reward of 1.0.

        TODO: Implement once answer comparison is done.
        """
        pytest.skip("TODO: Test correct answer reward once implemented")

    def test_reward_in_range(self, well_formed_completion):
        """Answer reward is in [0, 1]."""
        r = answer_reward(prompt="", completion=well_formed_completion)
        assert 0.0 <= r <= 1.0
