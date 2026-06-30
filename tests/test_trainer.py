"""
tests/test_trainer.py
=====================
Tests for src/trainer.py — GRPOExperimentTrainer wrapper.
"""

from __future__ import annotations

import pytest

from src.trainer import GRPOExperimentTrainer


class TestGRPOExperimentTrainer:
    def test_instantiation(self, experiment_config):
        """Trainer can be instantiated from a valid config."""
        trainer = GRPOExperimentTrainer(experiment_config)
        assert trainer.cfg.condition_id == "test_baseline"
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None

    def test_is_ready_false_before_setup(self, experiment_config):
        """is_ready is False before model and trainer are loaded."""
        trainer = GRPOExperimentTrainer(experiment_config)
        assert trainer.is_ready is False

    def test_load_model_raises_not_implemented(self, experiment_config):
        """load_model() raises NotImplementedError (skeleton)."""
        trainer = GRPOExperimentTrainer(experiment_config)
        with pytest.raises(NotImplementedError):
            trainer.load_model()

    def test_setup_trainer_raises_if_no_model(self, experiment_config):
        """setup_trainer() raises RuntimeError if model not loaded."""
        trainer = GRPOExperimentTrainer(experiment_config)
        with pytest.raises(RuntimeError):
            trainer.setup_trainer(train_dataset=None)

    def test_train_raises_if_no_trainer(self, experiment_config):
        """train() raises RuntimeError if trainer not set up."""
        trainer = GRPOExperimentTrainer(experiment_config)
        with pytest.raises(RuntimeError):
            trainer.train()

    def test_reward_fn_is_callable(self, experiment_config):
        """The reward function is a callable."""
        trainer = GRPOExperimentTrainer(experiment_config)
        assert callable(trainer._reward_fn)

    def test_save_not_implemented(self, experiment_config):
        """save() raises NotImplementedError (skeleton)."""
        trainer = GRPOExperimentTrainer(experiment_config)
        with pytest.raises(NotImplementedError):
            trainer.save()

    def test_load_not_implemented(self, experiment_config):
        """load() raises NotImplementedError (skeleton)."""
        trainer = GRPOExperimentTrainer(experiment_config)
        with pytest.raises(NotImplementedError):
            trainer.load("some/path")
