"""
src/evaluation.py
=================
Post-training evaluation pipeline.

Runs systematic evaluation of trained checkpoints across all conditions,
computes all metrics, and saves results to disk.

Usage:
    from src.evaluation import EvaluationPipeline
    from src.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/c1_baseline.yaml")
    pipeline = EvaluationPipeline(cfg)
    results = pipeline.run(checkpoint_path="outputs/c1_baseline/checkpoint-final")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import ExperimentConfig, GenerationConfig
from src.metrics import MetricsResult, compute_all_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Container for evaluation results for a single checkpoint.

    Attributes:
        condition_id: Experimental condition identifier.
        checkpoint_path: Path to the evaluated checkpoint.
        metrics: Computed MetricsResult.
        completions_sample: A sample of raw model completions for qualitative review.
        error: Error message if evaluation failed.
    """

    condition_id: str
    checkpoint_path: str
    metrics: Optional[MetricsResult] = None
    completions_sample: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether evaluation completed without errors."""
        return self.error is None and self.metrics is not None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationPipeline:
    """Post-training evaluation pipeline.

    Loads a checkpoint, generates completions on an eval dataset,
    computes all metrics, and saves results.

    Attributes:
        cfg: Full experiment configuration.
        generation_cfg: Generation parameters used during eval.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
    ) -> None:
        """Initialise the evaluation pipeline.

        Args:
            cfg: Full ExperimentConfig for the condition being evaluated.
        """
        self.cfg = cfg
        self.generation_cfg: GenerationConfig = cfg.generation
        logger.info(
            "EvaluationPipeline initialised — condition: %s", cfg.condition_id
        )

    def run(
        self,
        checkpoint_path: str,
        eval_dataset: Optional[Any] = None,
        k: int = 8,
        output_dir: Optional[str] = None,
    ) -> EvaluationResult:
        """Run the full evaluation pipeline on a checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint directory.
            eval_dataset: HuggingFace Dataset to evaluate on.
                If None, loads from cfg.dataset.
            k: Number of samples for pass@k computation.
            output_dir: Where to save evaluation results. Defaults to
                cfg.training.output_dir / "eval".

        Returns:
            An EvaluationResult with computed metrics.

        TODO:
            - Load model from checkpoint_path via trainer.load().
            - Load eval_dataset if not provided.
            - Call generate_completions() for each problem.
            - Call compute_all_metrics() on the completions.
            - Save results via save_results().
        """
        logger.info(
            "Running evaluation on checkpoint: %s", checkpoint_path
        )
        # TODO: Implement full evaluation pipeline
        raise NotImplementedError(
            "EvaluationPipeline.run() not yet implemented. See TODOs."
        )

    def generate_completions(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        k: int = 8,
    ) -> list[list[str]]:
        """Generate k completions per prompt using the loaded model.

        Args:
            model: Loaded language model.
            tokenizer: Loaded tokenizer.
            prompts: List of input prompts.
            k: Number of completions per prompt.

        Returns:
            A list of lists: for each prompt, k completion strings.

        TODO:
            - Batch prompts for efficient GPU utilisation.
            - Respect generation_cfg parameters (temperature, top_p, etc.).
            - Handle OOM by reducing batch size dynamically.
        """
        logger.info(
            "Generating %d completions for %d prompts.", k, len(prompts)
        )
        # TODO: Implement generation loop using src.generation
        raise NotImplementedError(
            "generate_completions() not yet implemented."
        )

    def save_results(
        self,
        result: EvaluationResult,
        output_dir: str,
    ) -> None:
        """Save evaluation results to disk.

        Saves:
          - metrics.json      : All scalar metrics
          - completions.jsonl : Sample completions for qualitative review
          - summary.txt       : Human-readable summary

        Args:
            result: The EvaluationResult to save.
            output_dir: Directory to save files to.

        TODO:
            - Implement JSON serialisation and file writing.
            - Append results to a cross-condition summary CSV.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Saving evaluation results to %s", out)
        # TODO: Save metrics.json, completions.jsonl, summary.txt
        raise NotImplementedError("save_results() not yet implemented.")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-condition comparison
# ─────────────────────────────────────────────────────────────────────────────


def compare_conditions(
    results: list[EvaluationResult],
) -> dict[str, Any]:
    """Compare evaluation results across experimental conditions.

    Produces a summary table for paper reporting.

    Args:
        results: List of EvaluationResults, one per condition.

    Returns:
        Dict mapping condition_id to metrics dict.

    TODO:
        - Build a pandas DataFrame with all metrics.
        - Compute relative changes vs. C1 baseline.
        - Export to CSV and LaTeX table format.
    """
    logger.info("Comparing %d conditions.", len(results))
    # TODO: Implement cross-condition comparison
    comparison: dict[str, Any] = {}
    for r in results:
        if r.success and r.metrics is not None:
            comparison[r.condition_id] = r.metrics.to_dict()
    return comparison
