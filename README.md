# Reward Hacking Guardrails and Exploration in GRPO

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-orange.svg)](https://peps.python.org/pep-0008/)
[![Status: Research](https://img.shields.io/badge/status-research-informational.svg)]()

> **Workshop Paper:** *Do Reward Hacking Guardrails Reduce Exploration Diversity in Group Relative Policy Optimization?*

---

## Overview

This repository contains the reproducible research code for a study on **reward hacking and exploration trade-offs in GRPO**. We systematically evaluate whether standard guardrail mechanisms (KL regularization, length caps) suppress exploration diversity when models are trained with hackable reward signals.

**Model:** Qwen2.5-1.5B-Instruct with LoRA fine-tuning via HuggingFace TRL GRPOTrainer  
**Tasks:** Countdown (primary), GSM8K (secondary)

---

## Research Question

> *Do reward hacking guardrails reduce exploration diversity in GRPO?*

We study six experimental conditions:

| ID  | Condition | Reward | Guardrail |
|-----|-----------|--------|-----------|
| C1  | Baseline | Answer correctness only | None |
| C2  | Hackable | Answer + length bonus + format | None |
| C3  | KL-Low | Hackable | KL regularization (β=0.01) |
| C4  | KL-Med | Hackable | KL regularization (β=0.05) |
| C5  | KL-High | Hackable | KL regularization (β=0.1) |
| C6  | LengthCap | Hackable | Hard length cap |

---

## Repository Structure

```
grpo-reward-hacking/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata and tooling config
├── .gitignore                   # Git ignore rules
│
├── configs/                     # YAML experiment configurations
│   ├── base.yaml                # Shared base config
│   ├── c1_baseline.yaml         # Condition 1: Baseline reward
│   ├── c2_hackable.yaml         # Condition 2: Hackable reward
│   ├── c3_kl_low.yaml           # Condition 3: KL β=0.01
│   ├── c4_kl_med.yaml           # Condition 4: KL β=0.05
│   ├── c5_kl_high.yaml          # Condition 5: KL β=0.1
│   └── c6_length_cap.yaml       # Condition 6: Hard length cap
│
├── src/                         # Core library code
│   ├── __init__.py
│   ├── config.py                # Pydantic/dataclass config models
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── prompts.py               # Prompt templates
│   ├── reward_functions.py      # Reward function interfaces
│   ├── metrics.py               # Evaluation metrics
│   ├── trainer.py               # GRPOTrainer wrapper
│   ├── evaluation.py            # Post-training evaluation
│   ├── generation.py            # Text generation utilities
│   ├── logging_utils.py         # W&B / TensorBoard / CSV logging
│   └── utils.py                 # General utilities (seed, GPU, FS)
│
├── scripts/                     # Entry-point scripts
│   ├── train.py                 # Launch training for a condition
│   ├── evaluate.py              # Run evaluation on checkpoints
│   ├── run_all.sh               # Run all conditions sequentially
│   └── kaggle_setup.sh          # Kaggle environment setup
│
├── notebooks/                   # Exploratory analysis notebooks
│   ├── 00_data_exploration.ipynb
│   ├── 01_reward_analysis.ipynb
│   └── 02_results_visualization.ipynb
│
├── tests/                       # pytest test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_prompts.py
│   ├── test_reward_functions.py
│   ├── test_metrics.py
│   ├── test_trainer.py
│   ├── test_evaluation.py
│   ├── test_generation.py
│   ├── test_logging_utils.py
│   └── test_utils.py
│
├── outputs/                     # Experiment outputs (gitignored)
│   └── .gitkeep
│
└── docs/                        # Documentation
    ├── installation.md
    ├── repository_overview.md
    └── experiment_roadmap.md
```

---

## Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/<your-org>/grpo-reward-hacking.git
cd grpo-reward-hacking

# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Kaggle Setup

```bash
# Run the Kaggle setup script
bash scripts/kaggle_setup.sh
```

Or in a Kaggle notebook cell:

```python
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"])
```

### Running an Experiment

```bash
# Train baseline (C1)
python scripts/train.py --config configs/c1_baseline.yaml

# Train hackable (C2)
python scripts/train.py --config configs/c2_hackable.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --config configs/c1_baseline.yaml --checkpoint outputs/c1_baseline/checkpoint-final
```

---

## Configuration

All experiments are driven by YAML configs. Override any field via CLI:

```bash
python scripts/train.py --config configs/c1_baseline.yaml \
    training.max_steps=1000 \
    training.seed=42 \
    logging.wandb_project=my-project
```

See [`docs/repository_overview.md`](docs/repository_overview.md) for the full config schema.

---

## Metrics

| Metric | Description |
|--------|-------------|
| `pass@1` | Fraction of problems solved on first attempt |
| `pass@k` | Fraction solved within k samples |
| `unique_solution_count` | Diversity of correct solution strategies |
| `embedding_variance` | Semantic diversity of generated reasoning |
| `reasoning_length_stats` | Mean/std/max of `<think>` block lengths |
| `exploration_gap` | `pass@k − pass@1` (proxy for exploration) |

---

## Citation

```bibtex
@article{TODO_citation,
  title   = {Do Reward Hacking Guardrails Reduce Exploration Diversity in GRPO?},
  author  = {TODO},
  journal = {TODO Workshop},
  year    = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).