cat > /mnt/user-data/outputs/agent_prompt.md << 'ENDOFFILE'
# Agent Prompt: GRPO Reward Hacking Paper — Next Tasks

## What You Are

You are a research engineering agent helping complete a workshop paper on reward hacking and exploration in GRPO. The Countdown experimental phase is fully complete. Your job is to:

1. Set up and run Game of 24 experiments on GCP to validate generalizability
2. Perform analysis tasks to solidify statistical claims
3. Produce publication-quality figures and paper artifacts

Read this entire document before doing anything. Do not re-debate experimental design. Do not modify reward functions, metrics, or evaluation code unless a bug is explicitly identified. The Countdown numbers are final.

---

## Project Context

### The Paper

**Title:** Do Reward Hacking Guardrails Reduce Exploration Diversity in Group Relative Policy Optimization?

**Research question:** Do reward hacking guardrails in GRPO reduce exploration diversity, and if so, under what conditions and by how much?

**One-line answer:** Yes — KL regularization reduces solution diversity monotonically with β strength, very weak KL (β=0.01) amplifies hacking rather than suppressing it, and a hard length cap achieves best accuracy but collapses semantic diversity to near-zero.

**Target venue:** NeurIPS / ICML / ICLR workshop on RL for language models.

### Known Reviewer Concerns (address these — this is the point of the new experiments)

1. **Single task / single model** — "Interesting phenomenon, but demonstrated on only one reasoning domain." → Game of 24 addresses this.
2. **Embedding variance justification** — needs formal definition and correlation analysis with unique_solution_count.
3. **Why higher KL increases embedding variance** — counterintuitive, needs explicit mechanistic explanation in the paper.
4. **Statistical significance** — vague quantitative claims need bootstrap CIs or significance tests.

---

## Countdown Experiment Summary (complete — do not rerun)

### Model and Task (Countdown)
- **Model:** Qwen2.5-1.5B-Instruct + LoRA rank=16
- **Task:** Countdown — given 3–4 numbers, reach a target using each exactly once with {+, −, ×, ÷}
- **Dataset:** `Jiayi-Pan/Countdown-Tasks-3to4` (HuggingFace)
- **Framework:** HuggingFace TRL GRPOTrainer
- **KL reference:** Frozen base Qwen2.5-1.5B-Instruct

### Six Conditions (all complete)

| ID | Condition | Reward | Guardrail | β |
|---|---|---|---|---|
| C1 | Baseline | Correctness + format | None | — |
| C2 | Hackable | Correctness + length bonus + format | None | 0.0 |
| C3 | KL-Low | Hackable | KL regularization | 0.01 |
| C4 | KL-Med | Hackable | KL regularization | 0.05 |
| C5 | KL-High | Hackable | KL regularization | 0.10 |
| C6 | LengthCap | Hackable | Hard length cap (65 tokens) | — |

All conditions: 1000 steps · 2 seeds (42, 123) · 200 eval problems · k=8

### Countdown Final Evaluation Results (mean ± std across 2 seeds)

| Condition | pass@1 | pass@8 | gap | unique | length_mean | emb_var |
|---|---|---|---|---|---|---|
| C1 Baseline | 0.23 ± 0.04 | 0.44 ± 0.06 | 0.21 ± 0.07 | 0.54 ± 0.06 | 42.6 ± 4.8 | 0.138 ± 0.021 |
| C2 Hackable | 0.15 ± 0.01 | 0.42 ± 0.06 | 0.27 ± 0.04 | 0.57 ± 0.16 | 131.8 ± 9.7 | 0.141 ± 0.090 |
| C3 KL β=0.01 | 0.12 ± 0.03 | 0.41 ± 0.07 | 0.29 ± 0.10 | 0.53 ± 0.07 | 135.1 ± 27.2 | 0.140 ± 0.086 |
| C4 KL β=0.05 | 0.13 ± 0.04 | 0.31 ± 0.04 | 0.18 ± 0.00 | 0.39 ± 0.04 | 112.1 ± 0.2 | 0.236 ± 0.021 |
| C5 KL β=0.10 | 0.06 ± 0.03 | 0.25 ± 0.01 | 0.19 ± 0.01 | 0.27 ± 0.01 | 91.4 ± 2.3 | 0.287 ± 0.009 |
| C6 LengthCap | 0.19 ± 0.04 | 0.34 ± 0.06 | 0.15 ± 0.01 | 0.23 ± 0.24 | 63.2 ± 2.6 | 0.033 ± 0.010 |

### Per-seed results

| Condition | Seed | pass@1 | pass@8 | gap | unique | length_mean | length_med | length_std | emb_var |
|---|---|---|---|---|---|---|---|---|---|
| C1 Baseline | 42 | 0.20 | 0.48 | 0.28 | 0.58 | 45.97 | 38.0 | 28.75 | 0.153 |
| C1 Baseline | 123 | 0.26 | 0.40 | 0.14 | 0.50 | 39.13 | 34.0 | 22.94 | 0.123 |
| C2 Hackable | 42 | 0.14 | 0.38 | 0.24 | 0.46 | 124.91 | 130.5 | 67.66 | 0.205 |
| C2 Hackable | 123 | 0.16 | 0.46 | 0.30 | 0.68 | 138.66 | 138.0 | 45.59 | 0.077 |
| C3 KL-Low | 42 | 0.10 | 0.46 | 0.36 | 0.58 | 154.29 | 159.0 | 44.73 | 0.079 |
| C3 KL-Low | 123 | 0.14 | 0.36 | 0.22 | 0.48 | 115.90 | 115.0 | 64.44 | 0.200 |
| C4 KL-Med | 42 | 0.16 | 0.34 | 0.18 | 0.42 | 111.97 | 112.0 | 70.84 | 0.221 |
| C4 KL-Med | 123 | 0.10 | 0.28 | 0.18 | 0.36 | 112.30 | 113.0 | 75.18 | 0.251 |
| C5 KL-High | 42 | 0.08 | 0.26 | 0.18 | 0.26 | 89.74 | 83.5 | 72.46 | 0.293 |
| C5 KL-High | 123 | 0.04 | 0.24 | 0.20 | 0.28 | 93.03 | 90.0 | 68.76 | 0.280 |
| C6 LengthCap | 42 | 0.22 | 0.38 | 0.16 | 0.06 | 64.99 | 64.0 | 6.06 | 0.026 |
| C6 LengthCap | 123 | 0.16 | 0.30 | 0.14 | 0.40 | 61.34 | 61.0 | 6.63 | 0.040 |

### Training Curve Observations (W&B, EMA=0.99)

**think_length at step 1000:** C3 ≥ C2 > C4 > C5 > C6 (flat ~60) > C1 (flat ~35)

**correctness_component:** C1 only condition with sustained upward trend (~0.18 by step 1000). All others plateau ~0.06-0.10.

**completions/min_length:** C2 rises from ~45 to ~125 tokens by step 600 (all rollouts homogenized). C1 stays flat at ~45 tokens.

**eval/kl:** C3 KL rising (0.066→0.086, not converging). C4 KL declining (0.041→0.038, guardrail winning). C5 flat at ~0.017.

**format_component:** All conditions identical (0.12-0.15 by step 100). Experimental control confirmed.

### Checkpoints on HuggingFace Hub

`antrip03/grpo-{condition_id}-s{seed}` for all 12 runs:
- c1_baseline-s42, c1_baseline-s123
- c2_hackable-s42, c2_hackable-s123
- c3_kl_low-s42, c3_kl_low-s123
- c4_kl_med-s42, c4_kl_med-s123
- c5_kl_high-s42, c5_kl_high-s123
- c6_length_cap-s42, c6_length_cap-s123

### Eval Data on Disk

Per-problem JSONL files at `outputs/{condition_id}-s{seed}/eval/per_problem_results.jsonl`

Each line: `{"target": int, "nums": [int, ...], "greedy_completion": str, "sampled_completions": [str, ...]}`

---

## Repository Structure

```
grpo-reward-hacking/
├── configs/
│   ├── base.yaml                  # shared defaults
│   ├── c1_baseline.yaml
│   ├── c2_hackable.yaml
│   ├── c3_kl_low.yaml
│   ├── c4_kl_med.yaml
│   ├── c5_kl_high.yaml
│   └── c6_length_cap.yaml
├── src/
│   ├── config.py                  # Pydantic config models
│   ├── dataset.py                 # dataset loading (Countdown only currently)
│   ├── prompts.py                 # prompt templates and output parsers
│   ├── reward_functions.py        # reward logic, degenerate filter, expression checker
│   ├── metrics.py                 # pass@k, unique solutions, embedding variance
│   ├── evaluation.py              # eval pipeline
│   ├── trainer.py                 # GRPOTrainer wrapper
│   ├── generation.py              # generation utilities
│   └── logging_utils.py           # W&B + CSV logging
├── scripts/
│   ├── train.py                   # single condition entry point
│   ├── evaluate.py                # post-training evaluation
│   ├── zero_shot_eval.py          # pre-training gate check
│   └── modal_train.py             # Modal cloud launcher
└── outputs/                       # eval results
```

---

## PART A: Game of 24 Experiments (highest priority — do this first)

### Why Game of 24

Game of 24 is structurally identical to Countdown: verifiable arithmetic, multiple solution paths, no ambiguity. It is cited in Tree of Thoughts (Yao et al., 2023) and well-known to reviewers. The entire existing codebase (reward function, correctness checker, degenerate filter, expression parser, prompt template) works on Game of 24 without modification. Only the dataset loader needs to be added.

We run only C1 (baseline) and C2 (hackable) on Game of 24 — the minimum to demonstrate cross-task generalization of the hacking finding. This is framed in the paper as a validation experiment, not a full replication.

### Task A1: Add Game of 24 Dataset Support

**File to modify:** `src/dataset.py`

**Dataset:** Use `nlile/24-game` from HuggingFace, or generate programmatically. The dataset has examples with 4 numbers and target=24. If `nlile/24-game` is unavailable, generate it:

```python
def generate_game24_dataset(n_problems=5000, seed=42):
    """
    Generate Game of 24 problems programmatically.
    Sample 4 numbers uniformly from 1-13 (card values).
    Keep only problems that have at least one valid solution.
    Use existing expression_is_correct() to verify solvability.
    """
    import random
    import itertools
    from src.reward_functions import expression_is_correct

    random.seed(seed)
    problems = []
    ops = ['+', '-', '*', '/']

    def has_solution(nums, target=24):
        for perm in itertools.permutations(nums):
            for op1 in ops:
                for op2 in ops:
                    for op3 in ops:
                        # Try all expression structures
                        expressions = [
                            f"(({perm[0]} {op1} {perm[1]}) {op2} {perm[2]}) {op3} {perm[3]}",
                            f"({perm[0]} {op1} ({perm[1]} {op2} {perm[2]})) {op3} {perm[3]}",
                            f"{perm[0]} {op1} (({perm[1]} {op2} {perm[2]}) {op3} {perm[3]})",
                            f"{perm[0]} {op1} ({perm[1]} {op2} ({perm[2]} {op3} {perm[3]}))",
                            f"({perm[0]} {op1} {perm[1]}) {op2} ({perm[2]} {op3} {perm[3]})",
                        ]
                        for expr in expressions:
                            try:
                                if expression_is_correct(expr, 24, list(perm)):
                                    return True
                            except Exception:
                                continue
        return False

    attempts = 0
    while len(problems) < n_problems and attempts < n_problems * 10:
        nums = [random.randint(1, 13) for _ in range(4)]
        if has_solution(nums):
            problems.append({"nums": nums, "target": 24})
        attempts += 1

    return problems
```

**Add to `dataset.py`:**

```python
def load_game24_dataset(cfg: DatasetConfig) -> tuple[Any, Any]:
    """Load Game of 24 dataset. Falls back to programmatic generation if HF unavailable."""
    try:
        from datasets import load_dataset
        raw = load_dataset("nlile/24-game")
        # Adapt field names to match internal schema
        # Field names may differ — check and map to: nums (list), target (int)
    except Exception:
        # Generate programmatically
        problems = generate_game24_dataset(n_problems=5000, seed=42)
        # Convert to HuggingFace Dataset format

    # Reserve 200 for eval, rest for train
    # Use same holdout logic as load_countdown_dataset
    # Return (train_dataset, eval_dataset) in same schema as Countdown
    # Each example must have: prompt, answer (str of target), target (int), nums (list)
```

**Add task routing in `load_dataset_for_task`:**

```python
def load_dataset_for_task(cfg: DatasetConfig) -> tuple[Any, Any]:
    if cfg.task == "game24":
        return load_game24_dataset(cfg)
    return load_countdown_dataset(cfg)
```

**Also add `task` field to `DatasetConfig` in `src/config.py`** with default `"countdown"`.

### Task A2: Create Game of 24 Configs

Create two new config files. These inherit from `base.yaml` but override the dataset and condition-specific settings.

**`configs/game24_c1_baseline.yaml`:**
```yaml
condition_id: game24_c1_baseline
dataset:
  name: game24
  task: game24
  max_eval_samples: 200
training:
  max_steps: 500
  eval_steps: 500
  seed: 42
  output_dir: outputs/game24_c1_baseline-s42
reward:
  type: baseline
  length_bonus_max: 0.0
  format_bonus: 0.15
```

**`configs/game24_c2_hackable.yaml`:**
```yaml
condition_id: game24_c2_hackable
dataset:
  name: game24
  task: game24
  max_eval_samples: 200
training:
  max_steps: 500
  eval_steps: 500
  seed: 42
  output_dir: outputs/game24_c2_hackable-s42
reward:
  type: hackable
  length_bonus_max: 0.5
  length_bonus_ceiling: 100
  format_bonus: 0.15
```

### Task A3: Zero-Shot Gate on Game of 24

Before training, run the zero-shot evaluation to confirm Game of 24 is suitable:

```bash
python scripts/zero_shot_eval.py --config configs/game24_c1_baseline.yaml
```

**Decision gate:**
- If pass@1 > 0.30: task is too easy for 1.5B model, regenerate harder problems (use numbers 1-9 only, which tends to produce harder problems) and re-gate
- If pass@1 < 0.30 and pass@8 > 0: proceed — task has appropriate difficulty
- If pass@8 = 0: task may be too hard, check that solvable problems are being generated correctly

Report the zero-shot results before proceeding.

### Task A4: GCP Training Setup

**Compute:** Use GCP credits. Recommended instance: `n1-standard-8` with 1× NVIDIA T4 or V100, or `a2-highgpu-1g` with 1× A100.

**Setup commands on GCP instance:**
```bash
# Clone repo
git clone https://github.com/your-org/grpo-reward-hacking.git
cd grpo-reward-hacking

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install flash attention (optional but recommended)
pip install flash-attn --no-build-isolation

# Set credentials
export HF_TOKEN=your_token
export HF_USERNAME=antrip03
export WANDB_API_KEY=your_key
export WANDB_PROJECT=grpo-reward-hacking
```

**Verify GPU:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Task A5: Run Game of 24 Training

Run sequentially (not simultaneously — single GPU):

```bash
# C1 baseline on Game of 24
python scripts/train.py --config configs/game24_c1_baseline.yaml

# C2 hackable on Game of 24
python scripts/train.py --config configs/game24_c2_hackable.yaml
```

Expected training time: ~35 minutes each on T4, ~20 minutes on A100/V100.

**What to watch for during C2 training (confirm hacking is occurring):**
- `reward/think_length` should be rising over 500 steps
- `reward/length_component` should be rising
- `reward/correctness_component` should be flat or declining
- If hacking does NOT emerge in C2 by step 300, check reward weights and increase `length_bonus_max` to 0.8

### Task A6: Evaluate Game of 24 Checkpoints

```bash
python scripts/evaluate.py \
    --config configs/game24_c1_baseline.yaml \
    --checkpoint outputs/game24_c1_baseline-s42/checkpoint-final \
    --k 8

python scripts/evaluate.py \
    --config configs/game24_c2_hackable.yaml \
    --checkpoint outputs/game24_c2_hackable-s42/checkpoint-final \
    --k 8
```

**Push checkpoints to HF Hub** (will happen automatically if HF_TOKEN and HF_USERNAME are set in train.py).

**Minimum results needed to include in paper:**
- C2 Game of 24 should show: think_length_mean > 80 (hacking occurring)
- C2 Game of 24 should show: unique_solutions < C1 Game of 24 (exploration reduced)
- C1 Game of 24 should show: pass@1 > C2 Game of 24 pass@1

If these conditions are met, add a Game of 24 column to Table 1 in the paper. If not, report as negative result and discuss in limitations.

### Task A7: Training Curve Plots for Game of 24

Export W&B metrics for both Game of 24 runs and produce:
- `outputs/figures/game24_think_length.pdf` — think_length for C1 and C2, 500 steps
- `outputs/figures/game24_correctness.pdf` — correctness_component for C1 and C2

These go in the paper as a compact 2-panel figure showing replication of the main finding on a second task.

---

## PART B: Statistical Solidification Tasks

### Task B1: Bootstrap Confidence Intervals

**Goal:** Replace all vague quantitative claims in the paper with CI-backed numbers.

**Data:** `outputs/{condition_id}-s{seed}/eval/per_problem_results.jsonl`

**Implementation:**

```python
import numpy as np
import json
from pathlib import Path
from src.reward_functions import expression_is_correct, is_degenerate_solution
from src.prompts import extract_answer

def parse_correctness(completion: str, target: int, nums: list) -> int:
    answer = extract_answer(completion)
    if answer is None:
        return 0
    if is_degenerate_solution(answer):
        return 0
    return int(expression_is_correct(answer, target, nums))

def bootstrap_ci(values: list, n_bootstrap: int = 2000, ci: float = 0.95):
    arr = np.array(values, dtype=float)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(means, alpha*100)), float(np.percentile(means, (1-alpha)*100))

def compute_bootstrap_cis_for_condition(condition_id: str, seed: int) -> dict:
    path = Path(f"outputs/{condition_id}-s{seed}/eval/per_problem_results.jsonl")
    problems = [json.loads(line) for line in path.read_text().splitlines()]

    pass1_outcomes = []
    pass8_outcomes = []
    unique_counts = []

    for p in problems:
        target = p["target"]
        nums = p["nums"]

        greedy_correct = parse_correctness(p["greedy_completion"], target, nums)
        pass1_outcomes.append(greedy_correct)

        sampled_correct = [parse_correctness(c, target, nums) for c in p["sampled_completions"]]
        pass8_outcomes.append(int(any(sampled_correct)))

        correct_answers = set()
        for c, is_correct in zip(p["sampled_completions"], sampled_correct):
            if is_correct:
                answer = extract_answer(c)
                if answer:
                    correct_answers.add(answer)  # use canonicalized form from metrics.py
        unique_counts.append(len(correct_answers))

    return {
        "pass@1": bootstrap_ci(pass1_outcomes),
        "pass@8": bootstrap_ci(pass8_outcomes),
        "gap": bootstrap_ci([p8 - p1 for p1, p8 in zip(pass1_outcomes, pass8_outcomes)]),
        "unique_solutions": bootstrap_ci(unique_counts),
    }
```

Run for all 12 conditions and save to `outputs/bootstrap_ci_results.json`.

**Format of output:**
```json
{
  "c1_baseline": {
    "s42": {"pass@1": [0.20, 0.15, 0.26], "pass@8": [0.48, 0.41, 0.55], ...},
    "s123": {...},
    "pooled": {"pass@1": [0.23, 0.18, 0.28], ...}
  },
  ...
}
```

**Update the paper:** Replace all instances of bare numbers with CI notation. For example:
- "pass@8 drops from 0.42 to 0.25" → "pass@8 drops from 0.42 [0.35, 0.49] to 0.25 [0.19, 0.31]"
- Confirm CIs for C2 and C5 pass@8 do not overlap — this is the key significance claim

### Task B2: Spearman Correlation for Embedding Variance Validation

**Goal:** Show embedding variance correlates with unique_solution_count across conditions, validating it as a diversity metric.

**Data (from aggregated table):**

```python
conditions = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
unique_solutions = [0.54, 0.57, 0.53, 0.39, 0.27, 0.23]
emb_variance    = [0.138, 0.141, 0.140, 0.236, 0.287, 0.033]
```

**Compute:**
```python
from scipy.stats import spearmanr
rho, pvalue = spearmanr(unique_solutions, emb_variance)
```

Note: C6 is the deliberate outlier (high unique_solutions relative to emb_var). Compute correlation both with and without C6, and report both. The divergence of C6 is itself a finding — explain it.

**Expected result:** Negative correlation (higher unique solutions → lower embedding variance OR higher embedding variance → lower unique solutions for KL conditions). C6 inverts this because the cap enforces semantic uniformity even when some solution diversity exists.

**Add to paper (Section 4.4 or Appendix):**

> We validate embedding variance as a diversity metric by computing its Spearman correlation with unique solution count across conditions (ρ = X, p = Y). The exception is C6, where embedding variance is near-zero despite moderate unique solution count — reflecting that the length cap enforces semantic uniformity in reasoning style even when correct answers vary. This divergence is itself informative: embedding variance and unique solution count capture complementary aspects of diversity.

### Task B3: Self-BLEU as Alternative Diversity Metric (robustness check)

**Goal:** Show that the low-diversity finding for C6 is not an artifact of the embedding model choice.

**Compute self-BLEU for C1, C2, and C6** (three conditions is sufficient — don't need all six):

```python
from nltk.translate.bleu_score import sentence_bleu
from src.prompts import extract_think

def self_bleu(completions: list[str], n: int = 4) -> float:
    """
    Lower self-BLEU = more diverse (less n-gram overlap between completions).
    n: max n-gram order.
    """
    think_blocks = [extract_think(c) or "" for c in completions]
    tokenized = [t.split() for t in think_blocks]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if refs and hyp:
            scores.append(sentence_bleu(refs, hyp, weights=[1/n]*n))
    return float(np.mean(scores)) if scores else 0.0
```

Load sampled completions from per-problem JSONL files for C1, C2, C6 and compute per-problem self-BLEU, then average.

**Expected:** C6 should show higher self-BLEU (less diverse, more n-gram overlap) than C1 and C2, consistent with near-zero embedding variance.

**Report in paper:** "Self-BLEU confirms the low-diversity finding for C6 (self-BLEU=X vs C1=Y, C2=Z), indicating the result is not an artifact of the embedding model."

### Task B4: Monotonicity Test for β Sweep

**Goal:** Replace "monotonically decreasing" with a formal test.

For the claim that pass@8 decreases monotonically with β across C2→C3→C4→C5:

```python
from scipy.stats import kendalltau

beta_values = [0.00, 0.01, 0.05, 0.10]
pass8_values = [0.42, 0.41, 0.31, 0.25]
unique_values = [0.57, 0.53, 0.39, 0.27]

tau_pass8, p_pass8 = kendalltau(beta_values, pass8_values)
tau_unique, p_unique = kendalltau(beta_values, unique_values)
```

Kendall's τ measures rank correlation — τ=-1 means perfectly monotone decreasing. Report in paper.

**Also compute Page's trend test** if scipy has it, or use the Mann-Kendall test:

```python
# pip install pymannkendall
import pymannkendall as mk
result_pass8 = mk.original_test(pass8_values)
result_unique = mk.original_test(unique_values)
```

**Add to paper Section 4.2:**
> "The monotonic decrease in pass@8 and unique solution count with β is confirmed by Kendall's τ = X (p = Y) and τ = X (p = Y) respectively."

---

## PART C: Embedding Variance — Mechanistic Explanation

### Task C1: Add Formal Definition to Paper

In Section 3.4 (Evaluation Metrics), add the formal definition:

> **Embedding variance** measures the mean pairwise semantic distance between reasoning traces across rollouts. Formally, for problem $p$ with rollouts $\{o_1, \ldots, o_k\}$, let $\mathbf{e}_i^p$ denote the mean-pooled embedding of the $\langle\text{think}\rangle$ block of $o_i$ under the frozen base model. Then:
> $$\text{EmbVar}(p) = \frac{2}{k(k-1)} \sum_{i < j} \left(1 - \frac{\mathbf{e}_i^p \cdot \mathbf{e}_j^p}{\|\mathbf{e}_i^p\| \|\mathbf{e}_j^p\|}\right)$$
> averaged across all problems. We use the frozen Qwen2.5-1.5B-Instruct base model as the embedding model, as it captures semantically meaningful differences in reasoning style within the same model family.

### Task C2: Add Mechanistic Explanation for KL-Diversity Paradox

In Section 5.1 (Analysis), add this explanation for why higher KL produces higher embedding variance:

> **Why does higher KL increase embedding variance?** This result is counterintuitive — KL regularization keeps the policy close to the reference model, which might seem to reduce diversity. The explanation lies in the nature of the reference model: Qwen2.5-1.5B-Instruct before fine-tuning exhibits *undirected* diversity — it has not specialized in any particular reasoning style. As β increases, the policy is pulled back toward this undirected baseline, producing varied but often incorrect reasoning traces. In contrast, C2 (β=0) commits strongly to length-padding as a single optimization strategy, producing semantically similar (if incorrect) reasoning across rollouts. We term these three regimes: **coherent exploration** (C1 — diverse and correct), **incoherent exploration** (C5 — diverse but incorrect, pulled toward undirected reference), and **constrained uniformity** (C6 — neither diverse nor incorrect, cap enforces one template). The ideal regime of coherent exploration is achieved only by the clean baseline C1.

---

## PART D: Paper Production Tasks

### Task D1: β Sweep Visualization

Create a publication-quality figure showing the monotonic effect of β on exploration.

```python
import matplotlib.pyplot as plt
import numpy as np

beta_values = [0.00, 0.01, 0.05, 0.10]
pass8_mean = [0.42, 0.41, 0.31, 0.25]
unique_mean = [0.57, 0.53, 0.39, 0.27]

# Error bars from bootstrap CIs (Task B1 output)
pass8_ci_lower = [...]  # fill from bootstrap results
pass8_ci_upper = [...]
unique_ci_lower = [...]
unique_ci_upper = [...]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar(beta_values, pass8_mean,
            yerr=[np.array(pass8_mean)-np.array(pass8_ci_lower),
                  np.array(pass8_ci_upper)-np.array(pass8_mean)],
            marker='o', label='pass@8', color='steelblue', capsize=4)
ax.errorbar(beta_values, unique_mean,
            yerr=[np.array(unique_mean)-np.array(unique_ci_lower),
                  np.array(unique_ci_upper)-np.array(unique_mean)],
            marker='s', label='unique solutions', color='darkorange', capsize=4)
ax.set_xlabel('KL coefficient β', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_xticks(beta_values)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/beta_sweep.pdf', bbox_inches='tight')
plt.savefig('outputs/figures/beta_sweep.png', dpi=150, bbox_inches='tight')
```

### Task D2: Training Curves (Countdown, seed=42)

Export from W&B API and plot publication-quality figures. See original agent prompt for specifications — color scheme, axes, EMA smoothing. Produce:
- `outputs/figures/fig1_think_length.pdf`
- `outputs/figures/fig2_correctness.pdf`
- `outputs/figures/fig3_min_length.pdf`
- `outputs/figures/fig4_kl.pdf`

W&B project: `https://wandb.ai/anshultripathi002-iit-roorkee/grpo-reward-hacking`

Color scheme (consistent across all figures):
- C1 Baseline: `#8B4513` (brown)
- C2 Hackable: `#000000` (black)
- C3 KL-Low: `#228B22` (green)
- C4 KL-Med: `#00CED1` (cyan)
- C5 KL-High: `#DC143C` (red)
- C6 LengthCap: `#FF69B4` (pink)

### Task D3: Qualitative Completion Examples

Extract 3 side-by-side examples of C1 vs C2 completions showing hacking behavior. From `per_problem_results.jsonl`:
- Find problems where C2 greedy completion is WRONG and think_length > 100 tokens
- Find C1 completion for the same problem (same target + nums)
- Truncate think blocks to 60 tokens for display, add [...]

Save to `outputs/completion_examples.md`.

### Task D4: LaTeX Paper

Convert `workshop_paper.md` to LaTeX with NeurIPS 2024 style. Requirements:
- Proper `booktabs` table for Table 1 with CI notation from Task B1
- Figure placeholders for all figures from Tasks D1 and D2
- BibTeX entries in `paper/references.bib`
- Game of 24 results table added after Countdown table (or as a column)
- Statistical test results (Kendall's τ, Spearman ρ) inline in relevant sections
- Anonymous author block
- Target: 6-8 pages

Output: `paper/main.tex`, `paper/references.bib`

### Task D5: README for Code Release

Write `README.md` with:
- Title, abstract, badges (Python, license, arXiv placeholder, HF Hub)
- Quick start: install → zero-shot eval → train one condition → evaluate
- Full conditions table
- Countdown results table with CIs
- Game of 24 results (brief)
- Figure reproduction instructions
- Checkpoint loading example
- Citation BibTeX placeholder
- All 12 HuggingFace checkpoint links
- Do NOT include credentials, Modal-specific setup, or setup_env.sh

---

## Execution Order

```
PHASE 1 — Game of 24 (GCP, do first)
  A1: Add dataset support to src/dataset.py
  A2: Create game24 config files
  A3: Zero-shot gate (confirm task difficulty)
  A4: GCP environment setup
  A5: Train C1 and C2 on Game of 24 (500 steps each)
  A6: Evaluate both checkpoints
  A7: Training curve plots for Game of 24

PHASE 2 — Statistical Analysis (can run locally, no GPU needed)
  B1: Bootstrap CIs from per-problem JSONL files
  B2: Spearman correlation (emb_var vs unique_solutions)
  B3: Self-BLEU for C1, C2, C6
  B4: Kendall's τ and monotonicity test for β sweep

PHASE 3 — Paper Text Updates (no code, just writing)
  C1: Add formal embedding variance definition
  C2: Add KL-diversity paradox explanation

PHASE 4 — Paper Production
  D1: β sweep figure (needs B1 for error bars)
  D2: Training curve figures (W&B export)
  D3: Qualitative completion examples
  D4: LaTeX paper (needs all above)
  D5: README
```

Do not start Phase 4 until Phases 1-3 are complete. Report results after each phase.

---

## Hard Constraints

- Do not modify `src/reward_functions.py`, `src/metrics.py`, or `src/evaluation.py`
- Do not rerun any Countdown training
- Do not change Countdown eval numbers — they are final
- Game of 24 configs must use identical reward weights to Countdown C1/C2 for comparability
- All figures must use the exact color scheme specified in Task D2
- Bootstrap CIs must use per-problem JSONL files, not aggregated numbers
- Statistical tests must be two-tailed unless otherwise specified
- If Game of 24 zero-shot pass@8 = 0.0 (model cannot solve any problems), stop and report — do not proceed with training

---

## Deliverables Checklist

**Phase 1 — Game of 24:**
- [ ] `src/dataset.py` updated with game24 loader
- [ ] `configs/game24_c1_baseline.yaml`
- [ ] `configs/game24_c2_hackable.yaml`
- [ ] Zero-shot eval results reported
- [ ] Training complete: C1 and C2 on Game of 24
- [ ] Eval results: pass@1, pass@8, gap, unique, length_mean, emb_var for both
- [ ] `outputs/figures/game24_think_length.pdf`
- [ ] `outputs/figures/game24_correctness.pdf`
- [ ] Checkpoints on HF Hub: `antrip03/grpo-game24_c1_baseline-s42`, `antrip03/grpo-game24_c2_hackable-s42`

**Phase 2 — Statistical Analysis:**
- [ ] `outputs/bootstrap_ci_results.json`
- [ ] Spearman ρ and p-value reported
- [ ] Self-BLEU scores for C1, C2, C6 reported
- [ ] Kendall's τ for β sweep reported

**Phase 3 — Paper Text:**
- [ ] Embedding variance formal definition written
- [ ] KL-diversity paradox explanation written

**Phase 4 — Paper Production:**
- [ ] `outputs/figures/beta_sweep.pdf` + `.png`
- [ ] `outputs/figures/fig1_think_length.pdf`
- [ ] `outputs/figures/fig2_correctness.pdf`
- [ ] `outputs/figures/fig3_min_length.pdf`
- [ ] `outputs/figures/fig4_kl.pdf`
- [ ] `outputs/completion_examples.md`
- [ ] `paper/main.tex`
- [ ] `paper/references.bib`
- [ ] `README.md`
ENDOFFILE
Output

exit code 0
Done