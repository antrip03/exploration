from __future__ import annotations

from src.evaluation import EvaluationPipeline, EvaluationResult, compare_conditions
from src.metrics import MetricsResult


def test_evaluation_result_success_property():
    ok = EvaluationResult(condition_id="c1", checkpoint_path="ckpt", metrics=MetricsResult())
    assert ok.success is True
    failed = EvaluationResult(condition_id="c1", checkpoint_path="ckpt", error="boom")
    assert failed.success is False


def test_compare_conditions_filters_failed_results():
    results = [
        EvaluationResult(condition_id="c1", checkpoint_path="ckpt", metrics=MetricsResult(pass_at_1=0.2)),
        EvaluationResult(condition_id="c2", checkpoint_path="ckpt", error="oops"),
    ]
    comparison = compare_conditions(results)
    assert "c1" in comparison
    assert "c2" not in comparison


def test_save_results_writes_files(tmp_path, experiment_config):
    pipeline = EvaluationPipeline(experiment_config)
    result = EvaluationResult(
        condition_id="c1",
        checkpoint_path="ckpt",
        metrics=MetricsResult(pass_at_1=0.1, pass_at_k=0.4, k=8),
        per_problem=[{"target": 24}],
    )
    pipeline.save_results(result, str(tmp_path))
    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "per_problem_results.jsonl").exists()
