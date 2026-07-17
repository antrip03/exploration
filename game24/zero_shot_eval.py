"""Zero-shot gate evaluation for Game of 24 — run before training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.generation import generate_batch, generate_k_completions
from src.metrics import compute_all_metrics
from src.trainer import GRPOExperimentTrainer
from src.logging_utils import setup_python_logging
from src.prompts import build_chat_messages
from game24.dataset import load_game24_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a zero-shot Game of 24 gate.")
    parser.add_argument("--config", required=True, help="Path to the experiment config.")
    parser.add_argument("--k", type=int, default=8, help="Samples per prompt.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of evaluation examples to use. Defaults to 200 on CUDA and 50 on CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation. Larger values are much faster on GPU.",
    )
    return parser.parse_args()


def format_prompt(example: dict, tokenizer) -> dict:
    """Apply the model chat template to one preprocessed example."""
    messages = build_chat_messages(str(example["prompt"]))
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {**example, "prompt": formatted}


def main() -> None:
    args = parse_args()
    setup_python_logging(level="INFO")
    cfg = ExperimentConfig.from_yaml(args.config)
    _, eval_dataset = load_game24_dataset(cfg.dataset)
    default_max_samples = 200 if torch.cuda.is_available() else 50
    max_samples = args.max_samples if args.max_samples is not None else default_max_samples
    eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    trainer = GRPOExperimentTrainer(cfg)
    trainer.load_model()
    print(
        f"Running zero-shot Game24 eval on {len(eval_dataset)} examples "
        f"using {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )

    formatted = eval_dataset.map(
        lambda row: format_prompt(row, trainer.tokenizer),
        desc="Formatting zero-shot prompts",
    )
    prompts = [str(row["prompt"]) for row in formatted]
    targets = [str(row["target"]) for row in formatted]
    numbers = [list(row["nums"]) for row in formatted]

    greedy_cfg = cfg.generation.model_copy(update={"do_sample": False, "num_return_sequences": 1})
    sample_cfg = cfg.generation.model_copy(update={"do_sample": True, "temperature": 0.7})
    print("Generating greedy completions...")
    greedy = generate_batch(
        trainer.model,
        trainer.tokenizer,
        prompts,
        greedy_cfg,
        batch_size=min(args.batch_size, len(prompts)),
    )
    print(f"Generating {args.k} sampled completions per prompt...")
    sampled = generate_k_completions(
        trainer.model,
        trainer.tokenizer,
        prompts,
        sample_cfg,
        k=args.k,
        batch_size=min(args.batch_size, len(prompts)),
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
    print("\n" + "=" * 60)
    print("Zero-shot Game of 24 Evaluation Results")
    print("=" * 60)
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key:40s} {value:.4f}")
        else:
            print(f"  {key:40s} {value}")
    print("=" * 60)

    # Decision gate (from agent.md)
    if data["pass@8"] == 0.0:
        print("\n*** GATE FAILED: pass@8 = 0.0 — model cannot solve any Game24 problems. ***")
        print("    Check that solvable problems are being generated correctly.")
        print("    Do NOT proceed with training.")
    elif data["pass@1"] > 0.30:
        print("\n*** GATE: pass@1 > 0.30 — task may be too easy for 1.5B model. ***")
        print("    Recommendation: regenerate with numbers 1-9 only (harder problems).")
    elif data["pass@1"] < 0.30 and data["pass@8"] > 0:
        print("\n*** GATE PASSED: task has appropriate difficulty. Proceed with training. ***")
    else:
        print("\n*** GATE: unexpected result — review metrics manually. ***")


if __name__ == "__main__":
    main()