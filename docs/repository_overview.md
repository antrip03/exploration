# Repository Overview

This document describes the architecture and design decisions of the
`grpo-reward-hacking` research codebase.

---

## Design Principles

1. **Modularity** — each concern lives in a single file; no circular imports
2. **Config-driven** — all experiment parameters live in YAML + Pydantic; no hardcoded magic numbers
3. **Reproducibility** — all seeds are set explicitly; configs are saved with checkpoints
4. **Skeleton-first** — all interfaces are defined before implementation; TODOs are explicit
5. **Kaggle-compatible** — no OS-specific paths; GPU detection; checkpoint helpers

---

## Module Map

```
src/
├── config.py           ← Pydantic config classes (source of truth for all params)
├── prompts.py          ← Prompt templates + output parsing
├── dataset.py          ← Dataset loading (Countdown, GSM8K)
├── reward_functions.py ← Reward function interfaces (C1–C6)
├── metrics.py          ← Evaluation metrics (pass@k, exploration_gap, ...)
├── trainer.py          ← GRPOTrainer wrapper (lifecycle: load → train → eval → save)
├── evaluation.py       ← Post-training evaluation pipeline
├── generation.py       ← Inference utilities (batch generation, k-completions)
├── logging_utils.py    ← W&B / TensorBoard / CSV logging
└── utils.py            ← Seed, GPU, filesystem, checkpoint management
```

---

## Configuration System

All configs are **Pydantic v2 BaseModel** classes defined in `src/config.py`.

### Class hierarchy

```
ExperimentConfig
├── ModelConfig         model.name, dtype, attn_implementation
├── LoRAConfig          r, alpha, dropout, target_modules
├── TrainingConfig      max_steps, lr, num_generations, seed, ...
├── RewardConfig        type, weights, kl_beta, max_reasoning_tokens
├── GenerationConfig    max_new_tokens, temperature, top_p, ...
├── LoggingConfig       use_wandb, wandb_project, log_dir, ...
└── DatasetConfig       name, splits, max_samples, ...
```

### Config loading

```python
from src.config import ExperimentConfig
cfg = ExperimentConfig.from_yaml("configs/c1_baseline.yaml")
```

YAML files in `configs/` follow a `defaults: [base]` convention.
The `base.yaml` defines all defaults; condition-specific files only override changed fields.

---

## Reward Function Architecture

```
get_reward_fn(cfg)
    │
    ├─ RewardType.BASELINE    → reward_baseline(cfg)
    │                              └─ answer_reward()
    │
    ├─ RewardType.HACKABLE    → reward_hackable(cfg)
    │                              ├─ answer_reward()
    │                              ├─ format_reward()
    │                              └─ length_bonus()
    │
    └─ RewardType.GUARDRAILED → reward_guardrailed(cfg)
                                   ├─ answer_reward()
                                   ├─ format_reward()
                                   ├─ length_bonus()
                                   ├─ KL penalty  (if kl_beta > 0)
                                   └─ length_penalty() (if max_tokens set)
```

All reward functions follow the TRL GRPOTrainer signature:
`(prompts: list[str], completions: list[str], **kwargs) -> list[float]`

---

## Prompt Format

All tasks use a unified structured format:

```
[System prompt]

[Task-specific question]

<think>
...
</think>

<answer>
...
</answer>
```

The `<think>` block is the chain-of-thought reasoning chain.
The `<answer>` block contains the final answer only.

Parsing utilities in `src/prompts.py` extract these blocks reliably via regex.

---

## Metrics Architecture

| Metric | Source | Notes |
|--------|--------|-------|
| `pass@1` | `src/metrics.py` | Single-attempt correctness |
| `pass@k` | `src/metrics.py` | Chen et al. 2021 unbiased estimator |
| `unique_solution_count` | `src/metrics.py` | Surface-level diversity |
| `embedding_variance` | `src/metrics.py` | Semantic diversity via SentenceTransformers |
| `reasoning_length_stats` | `src/metrics.py` | Think-block length distribution |
| `exploration_gap` | `src/metrics.py` | `pass@k − pass@1` |

---

## Trainer Lifecycle

```
GRPOExperimentTrainer(cfg)
    │
    ├── load_model()        Load base model + apply LoRA
    ├── setup_trainer()     Build TRL GRPOTrainer with reward_fn + datasets
    ├── train()             Run training loop
    ├── evaluate()          Compute MetricsResult on eval set
    ├── save()              Save adapter + config to disk
    └── load()              Restore adapter from checkpoint
```

---

## Logging Architecture

`ExperimentLogger` wraps three backends:

| Backend | Config key | Output |
|---------|-----------|--------|
| W&B | `logging.use_wandb` | Online dashboard + artifact store |
| TensorBoard | `logging.use_tensorboard` | Local tensorboard logs |
| CSV | `logging.use_csv` | `{condition_id}_metrics.csv` |

All three are written to simultaneously during training via `logger.log(step, metrics)`.

---

## File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Condition config | `configs/c{N}_{name}.yaml` | `configs/c3_kl_low.yaml` |
| Checkpoint | `outputs/{condition_id}/checkpoint-{step}` | `outputs/c1_baseline/checkpoint-500` |
| Eval results | `outputs/{condition_id}/eval/metrics.json` | |
| Logs | `outputs/{condition_id}/logs/run.log` | |
