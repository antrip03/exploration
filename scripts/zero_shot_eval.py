"""Run a small zero-shot Countdown evaluation before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.dataset import load_dataset_for_task, format_prompt
from src.generation import generate_batch, generate_k_completions
from src.metrics import compute_all_metrics
from src.trainer import GRPOExperimentTrainer
from src.logging_utils import setup_python_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a zero-shot Countdown gate.")
    parser.add_argument("--config", required=True, help="Path to the experiment config.")
    parser.add_argument("--k", type=int, default=8, help="Samples per prompt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_python_logging(level="INFO")
    cfg = ExperimentConfig.from_yaml(args.config)
    _, eval_dataset = load_dataset_for_task(cfg.dataset)
    eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))

    trainer = GRPOExperimentTrainer(cfg)
    trainer.load_model()

    formatted = eval_dataset.map(
        lambda row: format_prompt(row, trainer.tokenizer),
        desc="Formatting zero-shot prompts",
    )
    prompts = [str(row["prompt"]) for row in formatted]
    targets = [str(row["target"]) for row in formatted]
    numbers = [list(row["nums"]) for row in formatted]

    greedy_cfg = cfg.generation.model_copy(update={"do_sample": False, "num_return_sequences": 1})
    sample_cfg = cfg.generation.model_copy(update={"do_sample": True, "temperature": 0.7})
    greedy = generate_batch(trainer.model, trainer.tokenizer, prompts, greedy_cfg, batch_size=1)
    sampled = generate_k_completions(
        trainer.model,
        trainer.tokenizer,
        prompts,
        sample_cfg,
        k=args.k,
        batch_size=1,
    )
    metrics = compute_all_metrics(
        completions_per_problem=sampled,
        ground_truths=targets,
        k=args.k,
        greedy_completions=greedy,
        numbers_per_problem=numbers,
        tokenizer=trainer.tokenizer,
    )
    data = metrics.to_dict()
    print(f"pass@1: {data['pass@1']:.4f}")
    print(f"pass@{args.k}: {data[f'pass@{args.k}']:.4f}")
    print(f"exploration_gap: {data['exploration_gap']:.4f}")
    if data["pass@1"] > 0.65:
        print("Recommendation: switch to a harder task or a smaller model.")
    else:
        print("Recommendation: Countdown looks suitable for the experiment.")


if __name__ == "__main__":
    main()
