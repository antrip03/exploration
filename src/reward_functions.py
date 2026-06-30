"""
src/reward_functions.py
=======================
Reward function interfaces for all experimental conditions.

Each reward function takes a list of (prompt, completion) pairs and returns
a list of scalar reward values. This matches the signature expected by
HuggingFace TRL GRPOTrainer.

Conditions:
  C1 — reward_baseline()      : answer correctness only
  C2 — reward_hackable()      : answer + length bonus + format
  C3–C5 — reward_guardrailed(): hackable + KL divergence penalty (beta sweep)
  C6 — reward_guardrailed()   : hackable + hard length cap

Component functions (composable):
  - answer_reward()    : checks correctness of <answer> block
  - format_reward()    : checks for well-formed <think>/<answer> tags
  - length_bonus()     : rewards longer reasoning (hackable signal)
  - length_penalty()   : penalises reasoning beyond a hard cap

Usage:
    from src.reward_functions import get_reward_fn
    reward_fn = get_reward_fn(cfg.reward)
    rewards = reward_fn(prompts, completions)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

from src.config import RewardConfig, RewardType
from src.prompts import parse_model_output

logger = logging.getLogger(__name__)

# Type alias: matches TRL GRPOTrainer reward_funcs signature
RewardFunction = Callable[[list[str], list[str]], list[float]]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def get_reward_fn(cfg: RewardConfig) -> RewardFunction:
    """Factory: return the appropriate reward function given the config.

    Args:
        cfg: RewardConfig specifying type, weights, and guardrail params.

    Returns:
        A callable reward function with signature (prompts, completions) -> rewards.

    Raises:
        ValueError: If cfg.type is not a recognised RewardType.

    Example:
        >>> reward_fn = get_reward_fn(cfg.reward)
        >>> rewards = reward_fn(prompts=["..."], completions=["..."])
    """
    logger.info("Building reward function: type=%s", cfg.type)
    if cfg.type == RewardType.BASELINE:
        return reward_baseline(cfg)
    elif cfg.type == RewardType.HACKABLE:
        return reward_hackable(cfg)
    elif cfg.type == RewardType.GUARDRAILED:
        return reward_guardrailed(cfg)
    else:
        raise ValueError(f"Unknown reward type: {cfg.type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Composite reward functions
# ─────────────────────────────────────────────────────────────────────────────


def reward_baseline(cfg: RewardConfig) -> RewardFunction:
    """Return the baseline reward function (answer correctness only).

    Condition C1. No hackable signals, no guardrails.

    Args:
        cfg: RewardConfig (only answer_reward_weight is used).

    Returns:
        A reward function: (prompts, completions) -> list[float].

    TODO:
        - Implement answer comparison logic in answer_reward().
        - Decide on exact-match vs. numeric equivalence checking.
        - Handle formatting tolerance (e.g., '24' vs '24.0').
    """
    w_ans = cfg.answer_reward_weight

    def _reward_fn(
        prompts: list[str],
        completions: list[str],
        answers: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            r_ans = answer_reward(prompt=prompt, completion=completion, answer=None)
            rewards.append(w_ans * r_ans)
        return rewards

    return _reward_fn


def reward_hackable(cfg: RewardConfig) -> RewardFunction:
    """Return the hackable reward function (answer + length bonus + format).

    Condition C2. Includes signals that can be exploited without correct reasoning.

    Args:
        cfg: RewardConfig with format_reward_weight and length_bonus_weight.

    Returns:
        A reward function: (prompts, completions) -> list[float].

    TODO:
        - Calibrate length_bonus() to avoid trivial exploitation.
        - Implement format_reward() for tag-presence checking.
        - Log per-component reward breakdown for analysis.
    """
    w_ans = cfg.answer_reward_weight
    w_fmt = cfg.format_reward_weight
    w_len = cfg.length_bonus_weight

    def _reward_fn(
        prompts: list[str],
        completions: list[str],
        answers: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            r_ans = answer_reward(prompt=prompt, completion=completion, answer=None)
            r_fmt = format_reward(completion=completion)
            r_len = length_bonus(completion=completion)
            total = w_ans * r_ans + w_fmt * r_fmt + w_len * r_len
            rewards.append(total)
        return rewards

    return _reward_fn


def reward_guardrailed(cfg: RewardConfig) -> RewardFunction:
    """Return the guardrailed reward function (hackable + KL or length cap).

    Conditions C3–C6. Wraps hackable reward with one or more guardrails.

    Guardrail selection:
      - If kl_beta > 0  : KL divergence penalty is active (C3–C5).
      - If max_reasoning_tokens is not None : Hard length cap active (C6).
      - Both can be combined (though not in the current study design).

    Args:
        cfg: Full RewardConfig including guardrail parameters.

    Returns:
        A reward function: (prompts, completions) -> list[float].

    TODO:
        - Implement KL penalty computation (requires reference model logprobs).
        - Integrate KL from TRL GRPOTrainer's built-in beta parameter vs.
          manual computation — decide which approach to use.
        - Implement hard length truncation / zero-reward-past-cap in length_penalty().
    """
    w_ans = cfg.answer_reward_weight
    w_fmt = cfg.format_reward_weight
    w_len = cfg.length_bonus_weight
    kl_beta = cfg.kl_beta
    max_tokens = cfg.max_reasoning_tokens

    def _reward_fn(
        prompts: list[str],
        completions: list[str],
        answers: Optional[list[str]] = None,
        kl_divergences: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            r_ans = answer_reward(prompt=prompt, completion=completion, answer=None)
            r_fmt = format_reward(completion=completion)
            r_len = length_bonus(completion=completion)
            hackable_total = w_ans * r_ans + w_fmt * r_fmt + w_len * r_len

            # KL guardrail
            kl_penalty = 0.0
            if kl_beta > 0.0 and kl_divergences is not None:
                # TODO: Verify sign convention (subtract KL from reward)
                kl_penalty = kl_beta * kl_divergences[i]

            # Hard length-cap guardrail
            len_pen = 0.0
            if max_tokens is not None:
                len_pen = length_penalty(
                    completion=completion,
                    max_tokens=max_tokens,
                )

            rewards.append(hackable_total - kl_penalty - len_pen)
        return rewards

    return _reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Component reward functions
# ─────────────────────────────────────────────────────────────────────────────


def answer_reward(
    prompt: str,
    completion: str,
    answer: Optional[str] = None,
) -> float:
    """Compute the answer-correctness reward for a single completion.

    Extracts the <answer> block and compares to ground truth.

    Args:
        prompt: The input prompt (may contain the ground-truth answer in metadata).
        completion: The model's completion string.
        answer: Ground-truth answer string. If None, tries to extract from prompt.

    Returns:
        Reward in [0.0, 1.0]. 1.0 = correct, 0.0 = incorrect / missing.

    TODO:
        - Implement numeric equivalence check (not just string equality).
        - Handle expression evaluation for Countdown (e.g., '(3+1)*2 == 8').
        - Handle GSM8K answer extraction (####-separated).
        - Consider partial credit scoring.
    """
    parsed = parse_model_output(completion)
    if not parsed.has_answer:
        return 0.0

    # TODO: Compare parsed.answer_block against ground-truth answer
    # For now, return 0.0 as placeholder
    return 0.0


def format_reward(completion: str) -> float:
    """Compute the formatting reward for a single completion.

    Awards credit for the presence and correct nesting of
    <think>...</think> and <answer>...</answer> blocks.

    Args:
        completion: The model's completion string.

    Returns:
        Reward in [0.0, 1.0]. 1.0 = perfectly formatted.

    TODO:
        - Implement graduated scoring:
            0.5 for either tag present, 1.0 for both.
        - Penalise reversed tag order (answer before think).
        - Consider whether empty blocks should be penalised.
    """
    parsed = parse_model_output(completion)
    # TODO: Implement graduated format scoring
    if parsed.is_well_formed:
        return 1.0
    elif parsed.has_think or parsed.has_answer:
        return 0.5
    return 0.0


def length_bonus(
    completion: str,
    target_length: int = 200,
    scale: float = 0.01,
) -> float:
    """Compute a think-length bonus reward (hackable signal).

    Awards increasing reward for longer <think> blocks. This is intentionally
    a hackable signal — models can exploit it by generating verbose but
    uninformative reasoning.

    Args:
        completion: The model's completion string.
        target_length: Word count at which bonus is maximised.
        scale: Controls how fast the bonus grows.

    Returns:
        Reward in [0.0, 1.0].

    TODO:
        - Replace word-count with actual token count using the tokenizer.
        - Decide between linear and sigmoidal bonus curves.
        - Calibrate target_length and scale against the training distribution.
    """
    parsed = parse_model_output(completion)
    think_len = parsed.think_token_count
    # TODO: Implement proper bonus curve (e.g., sigmoid or linear)
    # Placeholder: simple linear normalisation
    return min(1.0, scale * think_len)


def length_penalty(
    completion: str,
    max_tokens: int,
    penalty_per_excess_token: float = 0.01,
) -> float:
    """Compute a hard length-cap penalty (guardrail signal).

    Returns a positive penalty value that is subtracted from the total reward
    when the <think> block exceeds max_tokens.

    Args:
        completion: The model's completion string.
        max_tokens: Maximum allowed tokens in the <think> block.
        penalty_per_excess_token: Penalty magnitude per excess token.

    Returns:
        Non-negative penalty value (0.0 if within cap).

    TODO:
        - Replace word-count with actual tokenizer token count.
        - Decide: hard zero-reward cutoff vs. soft linear penalty.
        - Consider whether truncation should happen at generation time instead.
    """
    parsed = parse_model_output(completion)
    think_len = parsed.think_token_count
    excess = max(0, think_len - max_tokens)
    # TODO: Implement actual penalty function
    return excess * penalty_per_excess_token
