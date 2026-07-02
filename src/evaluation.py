"""Inference-only checkpoint evaluation with separate greedy and sampled passes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import ExperimentConfig
from src.generation import generate_batch, generate_k_completions
from src.metrics import MetricsResult, compute_all_metrics
from src.utils import save_json, save_jsonl

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    condition_id: str
    checkpoint_path: str
    metrics: Optional[MetricsResult] = None
    completions_sample: list[str] = field(default_factory=list)
    per_problem: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.metrics is not None


class EvaluationPipeline:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        checkpoint_path: str,
        eval_dataset: Optional[Any] = None,
        k: int = 8,
        output_dir: Optional[str] = None,
    ) -> EvaluationResult:
        from src.dataset import format_prompt, load_countdown_dataset
        from src.trainer import GRPOExperimentTrainer

        if eval_dataset is None:
            _, eval_dataset = load_countdown_dataset(self.cfg.dataset)

        wrapper = GRPOExperimentTrainer(self.cfg)
        wrapper.load(checkpoint_path)
        formatted = eval_dataset.map(
            lambda row: format_prompt(row, wrapper.tokenizer),
            desc="Applying chat template for evaluation",
        )
        prompts = [str(row["prompt"]) for row in formatted]
        targets = [str(row["target"]) for row in formatted]
        numbers = [list(row["nums"]) for row in formatted]

        greedy_cfg = self.cfg.generation.model_copy(
            update={"do_sample": False, "num_return_sequences": 1}
        )
        greedy = generate_batch(
            wrapper.model,
            wrapper.tokenizer,
            prompts,
            greedy_cfg,
            batch_size=1,
        )
        sampled = self.generate_completions(
            wrapper.model,
            wrapper.tokenizer,
            prompts,
            k=k,
        )
        metrics = compute_all_metrics(
            sampled,
            targets,
            k=k,
            greedy_completions=greedy,
            numbers_per_problem=numbers,
            tokenizer=wrapper.tokenizer,
        )
        per_problem = [
            {
                "target": target,
                "nums": nums,
                "prompt": prompt,
                "greedy_completion": greedy_completion,
                "sampled_completions": completions,
            }
            for target, nums, prompt, greedy_completion, completions in zip(
                targets, numbers, prompts, greedy, sampled
            )
        ]
        result = EvaluationResult(
            condition_id=self.cfg.condition_id,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            completions_sample=[item for group in sampled[:5] for item in group[:2]],
            per_problem=per_problem,
        )
        destination = output_dir or str(Path(self.cfg.training.output_dir) / "eval")
        self.save_results(result, destination)
        return result

    def generate_completions(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        k: int = 8,
    ) -> list[list[str]]:
        sample_cfg = self.cfg.generation.model_copy(
            update={"do_sample": True, "temperature": 0.7}
        )
        return generate_k_completions(
            model,
            tokenizer,
            prompts,
            sample_cfg,
            k=k,
            batch_size=1,
        )

    def save_results(self, result: EvaluationResult, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "condition_id": result.condition_id,
            "checkpoint_path": result.checkpoint_path,
            "success": result.success,
            "error": result.error,
            "metrics": result.metrics.to_dict() if result.metrics else None,
        }
        save_json(payload, out / "eval_results.json")
        save_json(payload, out / "metrics.json")
        save_jsonl(result.per_problem, out / "per_problem_results.jsonl")
        summary = "\n".join(
            [f"{key}: {value}" for key, value in (payload["metrics"] or {}).items()]
        )
        (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")


def run_evaluation(
    checkpoint_path: str,
    config: ExperimentConfig,
    eval_dataset: Optional[Any] = None,
    k: int = 8,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Functional entry point used by the CLI."""
    result = EvaluationPipeline(config).run(
        checkpoint_path=checkpoint_path,
        eval_dataset=eval_dataset,
        k=k,
        output_dir=output_dir,
    )
    if not result.success or result.metrics is None:
        raise RuntimeError(result.error or "Evaluation failed")
    return result.metrics.to_dict()


def compare_conditions(results: list[EvaluationResult]) -> dict[str, Any]:
    return {
        result.condition_id: result.metrics.to_dict()
        for result in results
        if result.success and result.metrics is not None
    }
