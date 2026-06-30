"""
tests/conftest.py
=================
Shared pytest fixtures for the GRPO reward hacking test suite.

All fixtures are available to all test modules automatically.
"""

from __future__ import annotations

import pytest

from src.config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    RewardConfig,
    TrainingConfig,
    DatasetName,
    RewardType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    """Minimal ModelConfig for unit tests (no GPU needed)."""
    return ModelConfig(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        dtype="float32",
        attn_implementation="eager",
        max_length=256,
    )


@pytest.fixture
def lora_config() -> LoRAConfig:
    """Minimal LoRAConfig for unit tests."""
    return LoRAConfig(r=4, alpha=8, dropout=0.0)


@pytest.fixture
def training_config(tmp_path) -> TrainingConfig:
    """Minimal TrainingConfig for unit tests."""
    return TrainingConfig(
        output_dir=str(tmp_path / "outputs"),
        max_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        bf16=False,
        fp16=False,
    )


@pytest.fixture
def baseline_reward_config() -> RewardConfig:
    """RewardConfig for the C1 baseline condition."""
    return RewardConfig(type=RewardType.BASELINE, answer_reward_weight=1.0)


@pytest.fixture
def hackable_reward_config() -> RewardConfig:
    """RewardConfig for the C2 hackable condition."""
    return RewardConfig(
        type=RewardType.HACKABLE,
        answer_reward_weight=1.0,
        format_reward_weight=0.3,
        length_bonus_weight=0.1,
    )


@pytest.fixture
def guardrailed_kl_config() -> RewardConfig:
    """RewardConfig for the C4 KL-guardrailed condition."""
    return RewardConfig(
        type=RewardType.GUARDRAILED,
        answer_reward_weight=1.0,
        format_reward_weight=0.3,
        length_bonus_weight=0.1,
        kl_beta=0.05,
    )


@pytest.fixture
def guardrailed_cap_config() -> RewardConfig:
    """RewardConfig for the C6 length-cap condition."""
    return RewardConfig(
        type=RewardType.GUARDRAILED,
        answer_reward_weight=1.0,
        format_reward_weight=0.3,
        length_bonus_weight=0.1,
        max_reasoning_tokens=50,
    )


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Minimal GenerationConfig for unit tests."""
    return GenerationConfig(max_new_tokens=64, do_sample=False, num_return_sequences=1)


@pytest.fixture
def logging_config(tmp_path) -> LoggingConfig:
    """LoggingConfig pointing to a tmp directory."""
    return LoggingConfig(
        use_wandb=False,
        use_tensorboard=False,
        use_csv=False,
        log_dir=str(tmp_path / "logs"),
    )


@pytest.fixture
def dataset_config() -> DatasetConfig:
    """Minimal DatasetConfig for unit tests."""
    return DatasetConfig(
        name=DatasetName.COUNTDOWN,
        max_train_samples=10,
        max_eval_samples=5,
    )


@pytest.fixture
def experiment_config(
    model_config,
    lora_config,
    training_config,
    baseline_reward_config,
    generation_config,
    logging_config,
    dataset_config,
) -> ExperimentConfig:
    """Full ExperimentConfig assembled from unit-test sub-configs."""
    return ExperimentConfig(
        condition_id="test_baseline",
        description="Test config",
        model=model_config,
        lora=lora_config,
        training=training_config,
        reward=baseline_reward_config,
        generation=generation_config,
        logging=logging_config,
        dataset=dataset_config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sample data fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_countdown_prompt() -> str:
    """A sample formatted Countdown prompt."""
    from src.prompts import build_countdown_prompt
    return build_countdown_prompt(target=24, numbers=[1, 2, 3, 4, 6, 8])


@pytest.fixture
def well_formed_completion() -> str:
    """A sample well-formed model completion."""
    return (
        "<think>\n"
        "I need to make 24 from 1, 2, 3, 4, 6, 8.\n"
        "Let me try: (4 - 1) * 8 = 24. Yes!\n"
        "</think>\n"
        "<answer>\n"
        "(4 - 1) * 8\n"
        "</answer>"
    )


@pytest.fixture
def malformed_completion() -> str:
    """A sample malformed model completion (no tags)."""
    return "The answer is 24 because 3 times 8 equals 24."


@pytest.fixture
def missing_answer_completion() -> str:
    """A completion with think block but no answer block."""
    return "<think>3 * 8 = 24</think>"
