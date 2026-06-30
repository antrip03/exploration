"""
tests/test_config.py
====================
Tests for src/config.py — configuration classes and validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    DatasetConfig,
    DatasetName,
    ExperimentConfig,
    GenerationConfig,
    LoRAConfig,
    LoggingConfig,
    ModelConfig,
    RewardConfig,
    RewardType,
    TrainingConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestModelConfig:
    def test_default_instantiation(self):
        """ModelConfig can be created with defaults."""
        cfg = ModelConfig()
        assert cfg.name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.dtype == "bfloat16"

    def test_valid_dtype(self):
        """Valid dtypes are accepted."""
        for dtype in ("bfloat16", "float16", "float32"):
            cfg = ModelConfig(dtype=dtype)
            assert cfg.dtype == dtype

    def test_invalid_dtype_raises(self):
        """Invalid dtype raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(dtype="int8")

    def test_max_length_positive(self):
        """max_length must be positive."""
        with pytest.raises(ValidationError):
            ModelConfig(max_length=0)


# ─────────────────────────────────────────────────────────────────────────────
# LoRAConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestLoRAConfig:
    def test_defaults(self):
        """LoRAConfig has sensible defaults."""
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.alpha == 32
        assert cfg.enabled is True

    def test_r_must_be_positive(self):
        """LoRA rank must be >= 1."""
        with pytest.raises(ValidationError):
            LoRAConfig(r=0)

    def test_target_modules_non_empty(self):
        """Target modules list should be non-empty by default."""
        cfg = LoRAConfig()
        assert len(cfg.target_modules) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TrainingConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_defaults(self):
        """TrainingConfig has sensible defaults."""
        cfg = TrainingConfig()
        assert cfg.max_steps == 500
        assert cfg.seed == 42

    def test_bf16_fp16_mutually_exclusive(self):
        """Both bf16 and fp16 enabled simultaneously raises ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(bf16=True, fp16=True)

    def test_learning_rate_positive(self):
        """Learning rate must be > 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

    def test_num_generations_positive(self):
        """num_generations must be >= 1."""
        with pytest.raises(ValidationError):
            TrainingConfig(num_generations=0)


# ─────────────────────────────────────────────────────────────────────────────
# RewardConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardConfig:
    def test_baseline_defaults(self):
        """Baseline reward config has correct defaults."""
        cfg = RewardConfig(type=RewardType.BASELINE)
        assert cfg.format_reward_weight == 0.0
        assert cfg.kl_beta == 0.0
        assert cfg.max_reasoning_tokens is None

    def test_kl_beta_non_negative(self):
        """kl_beta must be >= 0."""
        with pytest.raises(ValidationError):
            RewardConfig(kl_beta=-0.1)

    def test_max_tokens_positive_or_none(self):
        """max_reasoning_tokens must be positive or None."""
        with pytest.raises(ValidationError):
            RewardConfig(max_reasoning_tokens=0)

    def test_max_tokens_none_allowed(self):
        """max_reasoning_tokens=None is valid."""
        cfg = RewardConfig(max_reasoning_tokens=None)
        assert cfg.max_reasoning_tokens is None


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentConfig:
    def test_from_yaml_c1(self):
        """ExperimentConfig loads c1_baseline.yaml correctly.

        TODO: Implement once from_yaml() is fully functional.
        """
        pytest.skip("TODO: test from_yaml() with actual YAML loading")

    def test_to_dict_is_serialisable(self, experiment_config):
        """to_dict() returns a JSON-serialisable dict."""
        import json
        d = experiment_config.to_dict()
        # Should not raise
        json.dumps(d, default=str)
        assert "condition_id" in d

    def test_round_trip_yaml(self, experiment_config, tmp_path):
        """Config can be saved and reloaded from YAML.

        TODO: Implement once to_yaml() / from_yaml() are fully working.
        """
        pytest.skip("TODO: test YAML round-trip")

    def test_condition_id_preserved(self, experiment_config):
        """condition_id is preserved through to_dict()."""
        d = experiment_config.to_dict()
        assert d["condition_id"] == "test_baseline"
