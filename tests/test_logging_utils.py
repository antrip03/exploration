"""
tests/test_logging_utils.py
============================
Tests for src/logging_utils.py — experiment logging.
"""

from __future__ import annotations

import logging

import pytest

from src.logging_utils import ExperimentLogger, setup_python_logging


class TestSetupPythonLogging:
    def test_sets_log_level_info(self):
        """INFO level is set correctly."""
        setup_python_logging(level="INFO")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_sets_log_level_debug(self):
        """DEBUG level is set correctly."""
        setup_python_logging(level="DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_creates_log_file(self, tmp_path):
        """Log file is created when log_file is specified."""
        log_file = str(tmp_path / "test.log")
        setup_python_logging(level="INFO", log_file=log_file)
        import os
        assert os.path.exists(log_file)


class TestExperimentLogger:
    def test_instantiation(self, experiment_config):
        """ExperimentLogger can be instantiated."""
        exp_logger = ExperimentLogger(experiment_config)
        assert exp_logger.cfg.condition_id == "test_baseline"

    def test_setup_runs_without_error(self, experiment_config):
        """setup() completes without error when W&B / TB / CSV are disabled."""
        exp_logger = ExperimentLogger(experiment_config)
        exp_logger.setup()  # Should not raise

    def test_log_does_not_raise(self, experiment_config):
        """log() does not raise even when backends are not initialised."""
        exp_logger = ExperimentLogger(experiment_config)
        exp_logger.setup()
        exp_logger.log(step=1, metrics={"loss": 0.5, "reward": 0.8})  # Should not raise

    def test_finish_does_not_raise(self, experiment_config):
        """finish() does not raise even when backends are not initialised."""
        exp_logger = ExperimentLogger(experiment_config)
        exp_logger.setup()
        exp_logger.finish()  # Should not raise

    def test_wandb_not_initialised_when_disabled(self, experiment_config):
        """W&B run is None when use_wandb=False."""
        exp_logger = ExperimentLogger(experiment_config)
        exp_logger.setup()
        assert exp_logger._wandb_run is None

    def test_tb_writer_not_initialised_when_disabled(self, experiment_config):
        """TensorBoard writer is None when use_tensorboard=False."""
        exp_logger = ExperimentLogger(experiment_config)
        exp_logger.setup()
        assert exp_logger._tb_writer is None
