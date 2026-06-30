"""
tests/test_evaluation.py
========================
Tests for src/evaluation.py — post-training evaluation pipeline.
"""

from __future__ import annotations

import pytest

from src.evaluation import EvaluationPipeline, EvaluationResult, compare_conditions
from src.metrics import MetricsResult


class TestEvaluationResult:
    def test_success_false_when_no_metrics(self):
        """EvaluationResult.success is False when metrics is None."""
        result = EvaluationResult(
            condition_id="c1_baseline",
            checkpoint_path="outputs/c1_baseline",
        )
        assert result.success is False

    def test_success_true_with_metrics(self):
        """EvaluationResult.success is True when metrics is set."""
        result = EvaluationResult(
            condition_id="c1_baseline",
            checkpoint_path="outputs/c1_baseline",
            metrics=MetricsResult(),
        )
        assert result.success is True

    def test_error_present_on_failure(self):
        """EvaluationResult with error message is not successful."""
        result = EvaluationResult(
            condition_id="c1_baseline",
            checkpoint_path="outputs/c1_baseline",
            error="Model loading failed.",
        )
        assert result.success is False
        assert result.error is not None


class TestEvaluationPipeline:
    def test_instantiation(self, experiment_config):
        """Pipeline can be instantiated from a config."""
        pipeline = EvaluationPipeline(experiment_config)
        assert pipeline.cfg.condition_id == "test_baseline"

    def test_run_raises_not_implemented(self, experiment_config):
        """run() raises NotImplementedError (skeleton)."""
        pipeline = EvaluationPipeline(experiment_config)
        with pytest.raises(NotImplementedError):
            pipeline.run(checkpoint_path="outputs/c1_baseline/ckpt")

    def test_generate_completions_raises_not_implemented(self, experiment_config):
        """generate_completions() raises NotImplementedError (skeleton)."""
        pipeline = EvaluationPipeline(experiment_config)
        with pytest.raises(NotImplementedError):
            pipeline.generate_completions(None, None, ["prompt"], k=4)

    def test_save_results_raises_not_implemented(self, experiment_config, tmp_path):
        """save_results() raises NotImplementedError (skeleton)."""
        pipeline = EvaluationPipeline(experiment_config)
        result = EvaluationResult(
            condition_id="test",
            checkpoint_path="ckpt",
            metrics=MetricsResult(),
        )
        with pytest.raises(NotImplementedError):
            pipeline.save_results(result, str(tmp_path))


class TestCompareConditions:
    def test_empty_list_returns_empty_dict(self):
        """compare_conditions([]) returns an empty dict."""
        result = compare_conditions([])
        assert result == {}

    def test_successful_conditions_included(self):
        """Successful results are included in comparison."""
        results = [
            EvaluationResult(
                condition_id="c1_baseline",
                checkpoint_path="ckpt",
                metrics=MetricsResult(pass_at_1=0.6),
            ),
            EvaluationResult(
                condition_id="c2_hackable",
                checkpoint_path="ckpt",
                metrics=MetricsResult(pass_at_1=0.5),
            ),
        ]
        comparison = compare_conditions(results)
        assert "c1_baseline" in comparison
        assert "c2_hackable" in comparison

    def test_failed_conditions_excluded(self):
        """Failed results are excluded from comparison."""
        results = [
            EvaluationResult(
                condition_id="c1_baseline",
                checkpoint_path="ckpt",
                error="Something went wrong.",
            )
        ]
        comparison = compare_conditions(results)
        assert "c1_baseline" not in comparison
