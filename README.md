# Reward Hacking Guardrails and Exploration in GRPO

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the experiment code for studying whether reward-hacking guardrails in GRPO reduce exploration diversity.

The current setup is Countdown-only:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Trainer: Hugging Face TRL `GRPOTrainer`
- Adaptation: LoRA via `peft`
- Dataset: `Jiayi-Pan/Countdown-Tasks-3to4`

## Layout

- `configs/` experiment configs for C1 to C6
- `src/` library code for config, dataset loading, rewards, metrics, generation, trainer, evaluation, logging, and utilities
- `scripts/` launchers for training, evaluation, zero-shot gating, and Kaggle setup
- `tests/` unit tests for the main experiment plumbing
- `docs/` notes including the TRL audit

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/zero_shot_eval.py --config configs/c1_baseline.yaml
python scripts/train.py --config configs/c1_baseline.yaml
python scripts/evaluate.py --config configs/c1_baseline.yaml --checkpoint outputs/c1_baseline/checkpoint-final
```

For the full sweep, use either of these:

```bash
python scripts/run_all.py --dry-run
python scripts/run_all.py --configs configs/c1_baseline.yaml configs/c2_hackable.yaml
```

## Configuration

Each config file inherits from `configs/base.yaml`. Common overrides are supported on the CLI:

```bash
python scripts/train.py --config configs/c2_hackable.yaml training.max_steps=200 seed=123
python scripts/train.py --config configs/c4_kl_med.yaml training.max_steps=500 reward.length_bonus_max=0.5
```

The training script writes outputs under the directory specified by each config, so you can inspect checkpoints in `outputs/<condition>/` after each run.

## Metrics

- `pass@1`
- `pass@k`
- `exploration_gap`
- `unique_solution_count`
- `reasoning_length/*`
- `embedding_variance` is kept as a compatibility no-op and is always `0.0`

## Kaggle

```bash
bash scripts/kaggle_setup.sh
```

## License

MIT. See [LICENSE](LICENSE).
