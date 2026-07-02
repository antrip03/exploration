# Project Plan: Reward Hacking Guardrails and Exploration in GRPO

> **Type:** Workshop Paper (Research)
> **Target Venues:** NeurIPS RIGL/SATA · ICML · ICLR workshops on RL + LLMs
> **Compute Budget:** Zero cost — free-tier Kaggle/Colab (T4/A10 GPU)
> **Status:** Design complete · Implementation not started

---

## 1. Research Question

> **Do reward hacking guardrails in GRPO reduce exploration diversity, and if so, under what conditions and by how much?**

### Sub-questions
- Can reward hacking be reliably induced in a controlled GRPO setting with a misspecified proxy reward?
- Does KL regularization suppress hacking behavior, and at what cost to exploration diversity?
- Is there a β threshold below which guardrails help without meaningfully reducing exploration?
- Does a hard length cap produce a qualitatively different tradeoff than KL regularization?

---

## 2. Core Hypothesis

Guardrails designed to prevent reward exploitation (KL penalty, length caps) may inadvertently constrain the policy's ability to explore diverse solution strategies. Stronger guardrails → less hacking, but also less exploration. The **pass@k − pass@1 gap** is the primary signal for detecting this tradeoff.

---

## 3. Objectives

| # | Objective | Status |
|---|---|---|
| 1 | Demonstrate reward hacking in a controlled GRPO setting | Design done |
| 2 | Measure exploration in a principled, quantifiable way | Design done |
| 3 | Implement guardrails and measure their effect on hacking + exploration | Not started |
| 4 | Characterize the guardrail-exploration tradeoff curve across β values | Not started |

---

## 4. Experimental Setup

### 4.1 Model
- **Primary:** Qwen2.5-1.5B-Instruct + LoRA (via `peft`)
- **Fallback:** SmolLM2 (if Countdown task is too easy for Qwen at baseline)
- **Rationale:** Smallest model capable of meaningful arithmetic reasoning; fits on T4 16GB with LoRA

> ⚠️ **Pre-experiment check required:** Run zero-shot eval on 100 Countdown problems before committing. If pass@1 > 65%, switch to 4-to-5 number problems or SmolLM2. High baseline accuracy reduces the incentive to hack and shrinks the exploration gap signal.

### 4.2 Framework
- **RL Training:** HuggingFace TRL — `GRPOTrainer`
- **Parameter Efficiency:** `peft` LoRA
- **Logging:** `wandb` (free tier) + CSV fallback
- **Environment:** Kaggle Notebooks (free T4/A10)

### 4.3 Task
- **Primary:** Countdown (arithmetic reasoning)
  - Given N numbers, reach a target using {+, −, ×, ÷}, each number used exactly once
  - Dataset: `Jiayi-Pan/Countdown-Tasks-3to4` (HuggingFace)
  - Why: Multiple valid solution paths make exploration genuinely meaningful; verifiable reward; supports deliberate reward misspecification
- **Secondary:** GSM8K — **cut for MVP**; reinstate only if time/compute allows

### 4.4 How GRPO Works Here
TRL's GRPOTrainer handles the algorithm entirely. The implementation provides only:
1. **Reward functions** — one per condition
2. **GRPOConfig** — experimental knobs (β, num_generations, max_steps, etc.)
3. **Dataset pipeline** — prompt formatting

At each training step:
```
For each prompt batch:
  1. Sample G rollouts from active policy     (num_generations = 8)
  2. Score each rollout via reward_fn         → r_i
  3. Normalize within group                   → A_i = (r_i − mean(r)) / std(r)
  4. Compute KL(active policy ∥ ref model)    (ref = frozen base Qwen, no LoRA)
  5. Loss = −A_i · log_prob + β · KL
  6. Backprop → update active policy only
```

**Reference model:** Frozen Qwen2.5-1.5B-Instruct (base checkpoint, no LoRA adapters).
Set once at trainer initialization; never updated. KL measures drift from this neutral prior —
**not** from the hacking policy (C2). This applies to C3, C4, C5 only; C1, C2, C6 use β=0.

---

## 5. Reward Function Design

### 5.1 Reward Components

| Component | Formula | Conditions Active |
|---|---|---|
| Correctness | +1.0 if answer correct, else 0 | All (C1–C6) |
| Length bonus | +scaled bonus proportional to `<think>` token count (up to ceiling) | C2–C6 |
| Format bonus | +0.15 for well-formed `<think>…</think><answer>…</answer>` | All (C1–C6) |

> ⚠️ **Length and format bonuses must be computed and logged separately** in the reward function code, even though they're summed into a single return value. This enables post-hoc attribution of hacking to length vs format gaming.

### 5.2 Reward Weights (starting point — calibrate in C2 pilot)

```python
CORRECTNESS_WEIGHT = 1.0
LENGTH_BONUS_MAX   = 0.5    # at some token ceiling TBD from pilot
FORMAT_BONUS       = 0.15
```

The length bonus must be strong enough for hacking to dominate in C2, but not so strong that correctness collapses to zero. The pilot calibrates this balance.

### 5.3 Degenerate Solution Filter
Solutions matching patterns `n × 1`, `n + 0`, `n − 0`, `n / 1` receive **zero correctness reward**
regardless of mathematical validity. This is **not a stretch goal** — without it, pass@1 in C2 is
polluted by degenerate exploitation, undermining the core finding.

---

## 6. Experimental Conditions

| ID | Condition | Reward | Guardrail | β |
|---|---|---|---|---|
| C1 | Baseline | Correctness + format | None | — |
| C2 | Hackable | Correctness + length + format | None | 0.0 |
| C3 | KL-Low | Hackable | KL regularization | 0.01 |
| C4 | KL-Med | Hackable | KL regularization | 0.05 |
| C5 | KL-High | Hackable | KL regularization | 0.1 |
| C6 | LengthCap | Hackable | Hard length cap | — |

**All conditions:**
- Start from the same base Qwen2.5-1.5B-Instruct checkpoint
- Train from step 0 with guardrails active from the start (not introduced mid-run)
- Use the same random seed(s) — minimum 2 seeds
- KL reference for C3/C4/C5 = frozen base model (not C2 policy)

**Minimum viable experiment:** C1, C2, C4

---

## 7. Key Hyperparameters

| Parameter | Value | Status |
|---|---|---|
| `num_generations` (k) | 8 | Locked |
| `beta` values | {0.0, 0.01, 0.05, 0.1} | Locked |
| `max_steps` per condition | TBD from pilot | Pending |
| `length_cap` for C6 | TBD from pilot | Pending |
| Seeds | 2 minimum | Locked |
| Eval frequency | Every 100 steps | Locked |
| LoRA rank | 16 | Locked |
| Reward weights | Starting point above; calibrate in pilot | Pending |

---

## 8. Exploration Metrics

| Metric | How Measured | Priority |
|---|---|---|
| **pass@k − pass@1 gap** | k=8 rollouts; any-correct minus greedy accuracy | **Primary** |
| **pass@1** | Greedy decode accuracy | Core |
| **pass@k** | Fraction of problems with ≥1 correct in 8 samples | Core |
| **Unique solution count** | Deduplicate correct expressions per problem (commutative normalization) | Secondary |
| **Response length distribution** | Mean + std of `<think>` token count across rollouts | Hacking indicator |
| ~~Rollout embedding variance~~ | ~~Embed think traces, cosine distance~~ | **Cut** |

**Solution deduplication rule:** `(a+b)×c` and `c×(a+b)` are the same solution.
Normalize expressions for commutativity before deduplication.

---

## 9. Reward Hacking Indicators

- `<think>` block length increases monotonically over training steps
- pass@1 flat or declining while total reward increases
- Degenerate solutions (`n×1`, `n+0`) appearing at high frequency before filter kicks in
- pass@k gap collapsing — model converges to one exploit strategy
- High format reward, low correctness reward in logged component breakdown

---

## 10. Workflow

### Phase 0 — Pre-experiment Validation
```
[ ] Zero-shot eval: run Qwen2.5-1.5B-Instruct on 100 Countdown problems
      → If pass@1 > 65%: switch to 4-to-5 number problems or SmolLM2
[ ] Verify TRL GRPOTrainer source:
      → Confirm beta=0 fully disables KL (not just low-weights it)
      → Confirm KL direction is KL(policy∥ref), not KL(ref∥policy)
      → Confirm ref_model is base model without LoRA adapters
[ ] Lock remaining hyperparameters (steps, reward weights starting point)
[ ] Set up Kaggle environment: TRL + peft + wandb
[ ] Confirm W&B logging works end-to-end with a dummy run
```

### Phase 1 — C2 Pilot Run
```
[ ] Run C2 for ~300–500 steps
[ ] Confirm hacking signal: length ↑, accuracy flat, degenerate solutions appearing
[ ] Calibrate reward weights if signal too weak or too strong
[ ] Record: at what step does hacking clearly emerge?
      → This sets the minimum training horizon for all conditions
[ ] Record: what is the 90th percentile <think> token length at convergence?
      → This sets the length_cap threshold for C6
```

### Phase 2 — Full Experiment
```
[ ] Run all 6 conditions (or C1/C2/C4 MVP) from step 0
[ ] Guardrails active from step 0 for C3–C6
[ ] Checkpoint every 100 steps
[ ] At each checkpoint: evaluate pass@1, pass@k, length stats, unique solution count
[ ] Log to W&B per step: total reward, correctness component, length component,
    format component, KL divergence (C3–C5), mean think length
```

### Phase 3 — Analysis and Writeup
```
[ ] Plot: pass@k gap vs training steps for all conditions (primary figure)
[ ] Plot: mean <think> length vs pass@1 (hacking signature)
[ ] Plot: β sensitivity — exploration gap vs β at final checkpoint
[ ] Qualitative: sample rollouts from C2 vs C4 at same training step
[ ] Write paper
```

---

## 11. Implementation Status

### Done ✅
- Problem framing and research question
- Task selection (Countdown primary, GSM8K cut)
- Model and framework selection
- Reward function design (3 types, 6 conditions)
- Exploration metrics defined (embedding variance cut)
- Experimental grid designed
- Reference policy confirmed: frozen base model, not C2
- Training workflow confirmed: all conditions from step 0
- β values locked: {0.01, 0.05, 0.1}
- k locked: 8

### Decisions Still Needed 🔶
- [ ] Max training steps — from pilot
- [ ] Reward weight magnitudes — from pilot
- [ ] Length cap threshold for C6 — from pilot
- [ ] Solution deduplication normalization implementation

### Not Started ❌
- Environment setup
- Any code in `src/`
- Any training runs
- Analysis or writeup

---

## 12. Code Agent Instructions

> This section is the handoff to a code agent. It specifies **exactly what to check, change, and implement** in each file of the repository, given the design decisions finalized in this plan. Files are listed in implementation priority order.

---

### 12.0 Before Touching Any File — TRL Source Audit

**Do this first. It gates everything else.**

```python
# In a notebook or script, inspect TRL's GRPOTrainer source:
import inspect
from trl import GRPOTrainer
print(inspect.getsource(GRPOTrainer.compute_loss))
# or browse: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
```

Verify and document:
1. **KL direction:** is it `kl = policy_logprob - ref_logprob` (correct: KL(π∥ref)) or reversed?
2. **beta=0 behavior:** does `beta=0` fully zero out KL term, or is there a floor?
3. **ref_model handling:** when `ref_model` is passed explicitly, does TRL freeze it completely (no gradient, no LoRA)?
4. **reward function signature:** confirm exact expected signature — `(completions: list[str], **kwargs) -> list[float]` — and what kwargs are available (prompts, answers, etc.)
5. **num_generations:** confirm this is the G parameter (rollouts per prompt) and it maps to pass@k k

Write a short `docs/trl_audit.md` recording findings. If any of the above differ from expectation, the config and trainer wrapper must be adjusted before running anything.

---

### 12.1 `configs/base.yaml` — **MODIFY**

The base config needs to reflect all locked decisions. Check the existing file and ensure these fields are present and correct:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  # ref_model must be the SAME base checkpoint, not a trained checkpoint
  ref_model: "Qwen/Qwen2.5-1.5B-Instruct"

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]  # verify correct module names for Qwen2.5

training:
  num_generations: 8        # k for pass@k — locked
  max_steps: 1000           # placeholder; update after pilot
  eval_steps: 100           # checkpoint + eval frequency
  seed: 42                  # primary seed; run second seed=123 for all conditions
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4

reward:
  correctness_weight: 1.0
  length_bonus_max: 0.5     # calibrate in pilot
  length_bonus_ceiling: 512 # token count at which max bonus is reached; calibrate in pilot
  format_bonus: 0.15

dataset:
  name: "Jiayi-Pan/Countdown-Tasks-3to4"
  train_split: "train"
  eval_split: "test"
  max_eval_samples: 200     # for pass@k evaluation — enough for signal, fast enough to run

logging:
  wandb_project: "grpo-reward-hacking"
  log_reward_components: true   # must be true — log correctness/length/format separately
```

**What to check in existing file:**
- Is `ref_model` set? If missing, add it explicitly — do not rely on TRL default
- Is `num_generations` present? If it's set to anything other than 8, change it
- Is `log_reward_components` present? If missing, add — this is required for post-hoc hacking attribution
- Remove any GSM8K dataset references from base config

---

### 12.2 `configs/c1_baseline.yaml` through `configs/c6_length_cap.yaml` — **MODIFY**

Each condition config inherits from base and overrides only what differs. Check each file:

**c1_baseline.yaml**
```yaml
# inherits base
training:
  beta: 0.0   # confirm explicitly set, not absent (absence ≠ zero in all TRL versions)
reward:
  length_bonus_max: 0.0    # no length bonus in C1
  format_bonus: 0.15       # format bonus active in all conditions
```

**c2_hackable.yaml**
```yaml
training:
  beta: 0.0
reward:
  length_bonus_max: 0.5    # hackable — length bonus active
  format_bonus: 0.15
```

**c3_kl_low.yaml**
```yaml
training:
  beta: 0.01    # KL coefficient — confirm this maps to the beta param TRL expects
reward:
  length_bonus_max: 0.5
  format_bonus: 0.15
```

**c4_kl_med.yaml** — beta: 0.05

**c5_kl_high.yaml** — beta: 0.1

**c6_length_cap.yaml**
```yaml
training:
  beta: 0.0
  max_completion_length: TBD    # set after pilot; use 90th percentile of C2 think length
reward:
  length_bonus_max: 0.5
  format_bonus: 0.15
  hard_length_cap: true         # flag that triggers zero reward above threshold
```

**What to check across all condition configs:**
- β values match exactly: {0.0, 0.0, 0.01, 0.05, 0.1, 0.0} for C1–C6
- C5 is β=0.1 NOT β=0.2 (there was a discrepancy in earlier docs — 0.1 is correct per the image)
- No condition references GSM8K
- All conditions use same `num_generations: 8`

---

### 12.3 `src/reward_functions.py` — **REWRITE**

This is the most critical file. The existing implementation (if any) likely does not:
- Keep length and format components separable
- Implement the degenerate solution filter
- Return component breakdowns for logging

**Required interface:**

```python
def compute_reward_components(
    completion: str,
    answer: str,
    config: RewardConfig
) -> dict:
    """
    Returns individual reward components — never sum them here.
    Caller sums. This enables per-component logging.
    
    Returns:
        {
            "correctness": float,   # 1.0 or 0.0
            "length_bonus": float,  # 0.0 to config.length_bonus_max
            "format_bonus": float,  # 0.0 or config.format_bonus
            "is_degenerate": bool,  # True if trivial solution detected
            "think_length": int,    # token count of <think> block
        }
    """

def is_degenerate_solution(expression: str) -> bool:
    """
    Returns True for trivially correct but meaningless solutions.
    Patterns: n×1, n+0, n-0, n/1, 0+n, 1×n etc.
    Use regex — do not rely on model output formatting.
    Must handle both × and * as multiplication symbols.
    """

def length_bonus(think_tokens: int, max_bonus: float, ceiling: int) -> float:
    """
    Linear scale from 0 to max_bonus over 0 to ceiling tokens.
    Clamp at max_bonus above ceiling.
    Never negative.
    """

def format_bonus(completion: str, bonus_value: float) -> float:
    """
    Returns bonus_value if completion contains well-formed
    <think>...</think><answer>...</answer> structure, else 0.0.
    Use regex. Do not assume model follows format perfectly.
    """

# TRL-compatible wrappers — one per condition
def baseline_reward_fn(completions, answers, **kwargs) -> list[float]:
    """C1: correctness + format only"""

def hackable_reward_fn(completions, answers, **kwargs) -> list[float]:
    """C2–C6: correctness + length + format"""
    # Still apply degenerate filter — zero correctness for degenerate solutions
    # Log components to wandb inside this function if wandb is active
```

**Specific things to implement/fix:**

1. **Degenerate filter must run before correctness is awarded** — not as a post-process
2. **think_length must be extracted from `<think>` block only** — not total completion length. If no `<think>` block found, think_length = 0
3. **Length bonus uses token count, not character count** — use tokenizer to count, or approximate with `len(text.split()) * 1.3` if tokenizer not available in reward fn context
4. **All reward functions must return `list[float]`** — one float per completion, matching TRL's expected interface
5. **Log component breakdown per step** — call `wandb.log({"reward/correctness": ..., "reward/length": ..., "reward/format": ...})` inside the function, guarded by `if wandb.run is not None`
6. **Correctness check:** parse `<answer>` block, evaluate the arithmetic expression, compare to target. Use `sympy.simplify` or `eval()` with a safe parser — never raw `eval()` on model output

---

### 12.4 `src/metrics.py` — **REWRITE**

Existing file likely has stubs or missing implementations. Required:

```python
def pass_at_1(completions: list[str], answer: str) -> float:
    """Fraction of completions that are correct. Used greedy (k=1)."""

def pass_at_k(completions: list[str], answer: str, k: int = 8) -> float:
    """
    1.0 if any of k completions is correct, else 0.0.
    Averaged over problems at eval time.
    """

def exploration_gap(pass_k: float, pass_1: float) -> float:
    """pass@k - pass@1. Primary exploration metric."""

def unique_solution_count(correct_completions: list[str]) -> int:
    """
    Count distinct solution expressions among correct completions.
    Normalize for commutativity before deduplication:
      - Sort operands within each operation
      - Canonicalize operator symbols (* vs ×, / vs ÷)
    Returns 0 if no correct completions.
    """

def think_length_stats(completions: list[str]) -> dict:
    """
    Extract <think> block from each completion, measure token length.
    Returns {"mean": float, "std": float, "max": int, "min": int}
    If no <think> block found, length = 0 for that completion.
    """

def evaluate_checkpoint(
    model,
    tokenizer,
    dataset: list[dict],
    k: int = 8,
    num_problems: int = 200,
) -> dict:
    """
    Run full evaluation at a checkpoint.
    Samples k completions per problem, computes all metrics.
    Returns dict of all metrics for logging.
    """
```

**Specific things to check/fix:**
1. **`unique_solution_count` normalization:** must handle `a+b` vs `b+a` as the same. Implement a canonical form function — sort operands, standardize symbols
2. **pass@k must sample k completions with temperature > 0** — not greedy. Use `temperature=0.7` or similar. Greedy will return the same completion every time
3. **pass@1 should use greedy (temperature=0)** — not a sample. This is the exploitation measure
4. **Embedding variance is removed** — delete any existing implementation or stub

---

### 12.5 `src/config.py` — **MODIFY**

Check that the Pydantic/dataclass models reflect all current design decisions:

```python
@dataclass
class RewardConfig:
    correctness_weight: float = 1.0
    length_bonus_max: float = 0.5
    length_bonus_ceiling: int = 512   # tokens
    format_bonus: float = 0.15
    hard_length_cap: bool = False     # C6 flag
    hard_length_cap_tokens: int = 0   # set from pilot for C6

@dataclass  
class TrainingConfig:
    beta: float = 0.0                 # KL coefficient; 0.0 = no KL
    num_generations: int = 8          # k; must match pass@k k
    max_steps: int = 1000
    eval_steps: int = 100
    seed: int = 42
    # ref_model handled separately — always base Qwen, never a trained checkpoint
```

**What to check:**
- Is there a `gsm8k` or secondary task field anywhere? Remove it
- Is `beta` a first-class field? It should be, not buried in a generic `trainer_kwargs` dict
- Is `num_generations` present? Must be
- Is `hard_length_cap` and its threshold present for C6?
- Is `log_reward_components` flag present?

---

### 12.6 `src/dataset.py` — **MODIFY**

```python
def load_countdown_dataset(config: DatasetConfig) -> tuple[Dataset, Dataset]:
    """
    Load Jiayi-Pan/Countdown-Tasks-3to4 from HuggingFace.
    Returns (train_dataset, eval_dataset).
    Eval dataset capped at config.max_eval_samples (default 200).
    """

def format_prompt(example: dict, tokenizer) -> dict:
    """
    Format a Countdown problem into the Qwen chat template.
    System prompt must instruct model to use <think>...</think><answer>...</answer> format.
    Returns dict with "prompt" key as expected by GRPOTrainer.
    """
```

**What to check/fix:**
1. **Remove all GSM8K loading code** — or gate it behind a flag that defaults to False
2. **Prompt template must request the `<think>` format explicitly** — the reward function depends on it being present. If the model never produces `<think>` blocks, length bonus and format bonus both fail
3. **System prompt should NOT mention the reward structure** — the model shouldn't know it's being rewarded for length
4. **Verify the dataset field names:** check what `Jiayi-Pan/Countdown-Tasks-3to4` actually returns — field names like `nums`, `target`, `answer` need to be confirmed against the actual HuggingFace dataset card

---

### 12.7 `src/trainer.py` — **MODIFY**

This wraps GRPOTrainer. Key things to verify and fix:

```python
def build_trainer(config: ExperimentConfig, reward_fn) -> GRPOTrainer:
    """
    Constructs GRPOTrainer with correct ref_model and beta.
    
    Critical:
    - ref_model must be loaded separately from model, BEFORE LoRA is applied to model
    - ref_model must have requires_grad=False on all parameters
    - beta must come from config.training.beta — not hardcoded
    - reward_funcs must be a list: [reward_fn]
    """
```

**Specific things to implement:**

1. **Load ref_model before LoRA:** 
```python
# CORRECT order:
ref_model = AutoModelForCausalLM.from_pretrained(config.model.ref_model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

model = AutoModelForCausalLM.from_pretrained(config.model.name)
model = get_peft_model(model, lora_config)  # apply LoRA after ref is loaded
```

2. **Confirm beta=0 fully disables KL** — after the TRL source audit (12.0), handle the case where beta=0 still computes KL (wasteful on memory). If TRL doesn't short-circuit at beta=0, add a conditional to pass `ref_model=None` for C1/C2/C6

3. **Add checkpoint resume logic** — Kaggle sessions die. `GRPOConfig` should set `save_steps=100` and trainer should detect and resume from latest checkpoint:
```python
last_checkpoint = get_last_checkpoint(config.training.output_dir)
trainer.train(resume_from_checkpoint=last_checkpoint)
```

---

### 12.8 `src/evaluation.py` — **MODIFY**

Post-training evaluation runner. Must:

```python
def run_evaluation(
    checkpoint_path: str,
    config: ExperimentConfig,
    eval_dataset,
    k: int = 8,
) -> dict:
    """
    Load checkpoint, run pass@k evaluation over eval dataset.
    Saves results to outputs/<condition>/eval_results.json.
    Returns metrics dict.
    """
```

**What to check:**
1. **Loads LoRA checkpoint correctly** — use `PeftModel.from_pretrained`, not base model load
2. **Samples with temperature > 0 for pass@k** — must not use greedy for the k rollouts
3. **Saves per-problem results**, not just aggregates — needed for unique solution count analysis
4. **Does not re-run training** — evaluation is inference only; no gradient computation

---

### 12.9 `src/logging_utils.py` — **MODIFY**

```python
def log_step(step: int, reward_components: dict, metrics: dict):
    """
    Log per-step metrics to W&B and CSV.
    reward_components: {"correctness": float, "length": float, "format": float}
    metrics: {"pass@1": float, "pass@k": float, "gap": float, "think_length_mean": float}
    """

def log_eval(step: int, eval_results: dict):
    """Log full evaluation results at checkpoint steps."""
```

**What to check:**
1. **Reward components logged separately** — not just total reward. This is essential for detecting hacking (total reward ↑ while correctness ↓)
2. **`wandb` import guarded** — if wandb not configured, fall through to CSV only
3. **CSV backup always active** — do not make CSV logging conditional on wandb failing. Always write CSV. Kaggle sessions die and wandb sync can lag

---

### 12.10 `src/prompts.py` — **MODIFY**

```python
SYSTEM_PROMPT = """You are a mathematical reasoning assistant. 
Given a set of numbers and a target, find a way to reach the target 
using each number exactly once with basic arithmetic operations (+, -, *, /).

Show your reasoning in <think>...</think> tags, then give your answer in <answer>...</answer> tags.

Example:
<think>
I have numbers [3, 4, 5] and target 17.
3 + 4 = 7, 7 * ... hmm.
Let me try: 4 * 5 = 20, 20 - 3 = 17. Yes!
</think>
<answer>(4 * 5) - 3</answer>"""
```

**What to check:**
1. **Prompt does NOT mention reward** — never tell the model that longer reasoning is rewarded
2. **Format tags match exactly** what `reward_functions.py` and `metrics.py` parse — if reward parses `<answer>` with regex `<answer>(.*?)</answer>`, make sure the prompt instructs exactly that format
3. **Example in prompt uses valid Countdown-style solution** — don't use a GSM8K example
4. **Applied via Qwen's chat template** — use `tokenizer.apply_chat_template`, not string concatenation

---

### 12.11 `scripts/train.py` — **MODIFY**

Entry point for launching one condition. Must:

1. Accept `--config` path and optional overrides
2. Load config, build trainer, run training
3. **Print the condition ID and all key hyperparameters at launch** — so Kaggle output logs show exactly what ran
4. **Save a `run_config.json` to the output directory** at start of training — records the exact config used, for reproducibility
5. Handle checkpoint resume automatically (see 12.7)

---

### 12.12 `scripts/evaluate.py` — **MODIFY**

Standalone evaluation script. Must:
1. Accept `--checkpoint` path and `--config`
2. Run `evaluation.py::run_evaluation`
3. Print results table to stdout
4. Save to `outputs/<condition>/eval_results.json`

---

### 12.13 `scripts/kaggle_setup.sh` — **MODIFY**

```bash
#!/bin/bash
pip install trl peft wandb datasets transformers accelerate -q

# Verify GPU is available
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

# Verify TRL version — behavior differs across versions
python -c "import trl; print('TRL version:', trl.__version__)"

# Verify Qwen model can be loaded
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('Tokenizer loaded OK. Vocab size:', tok.vocab_size)
"
```

**What to add:**
- TRL version pin — specify exact version to avoid API drift between Kaggle runs
- `sympy` install if using for expression evaluation
- W&B login via env variable: `wandb login $WANDB_API_KEY`

---

### 12.14 `tests/test_reward_functions.py` — **IMPLEMENT (HIGH PRIORITY)**

These tests gate everything. Write before running any training:

```python
def test_degenerate_filter_trivial_multiply():
    assert is_degenerate_solution("5 * 1") == True
    assert is_degenerate_solution("1 * 5") == True

def test_degenerate_filter_trivial_add():
    assert is_degenerate_solution("5 + 0") == True
    assert is_degenerate_solution("0 + 5") == True

def test_degenerate_filter_valid_solution():
    assert is_degenerate_solution("(3 + 4) * 2") == False

def test_length_bonus_scales_linearly():
    assert length_bonus(0, max_bonus=0.5, ceiling=512) == 0.0
    assert length_bonus(256, max_bonus=0.5, ceiling=512) == pytest.approx(0.25)
    assert length_bonus(512, max_bonus=0.5, ceiling=512) == pytest.approx(0.5)
    assert length_bonus(1000, max_bonus=0.5, ceiling=512) == pytest.approx(0.5)  # clamped

def test_format_bonus_present():
    completion = "<think>some reasoning</think><answer>42</answer>"
    assert format_bonus(completion, 0.15) == pytest.approx(0.15)

def test_format_bonus_absent():
    completion = "42"
    assert format_bonus(completion, 0.15) == 0.0

def test_correctness_zero_for_degenerate():
    # Even if answer is mathematically correct, degenerate = 0 correctness
    components = compute_reward_components("5 * 1", answer="5", config=default_config)
    assert components["correctness"] == 0.0
    assert components["is_degenerate"] == True

def test_reward_components_are_separable():
    # Verify components are returned individually, not pre-summed
    components = compute_reward_components(valid_completion, answer="17", config=default_config)
    assert "correctness" in components
    assert "length_bonus" in components
    assert "format_bonus" in components
```

---

### 12.15 `tests/test_metrics.py` — **IMPLEMENT (HIGH PRIORITY)**

```python
def test_pass_at_k_any_correct():
    completions = ["wrong", "wrong", "correct", "wrong"]
    assert pass_at_k(completions, answer="5", k=4) == 1.0

def test_pass_at_k_none_correct():
    completions = ["wrong", "wrong", "wrong", "wrong"]
    assert pass_at_k(completions, answer="5", k=4) == 0.0

def test_exploration_gap():
    assert exploration_gap(pass_k=0.8, pass_1=0.6) == pytest.approx(0.2)

def test_unique_solution_count_deduplication():
    # a+b and b+a should be one unique solution
    solutions = ["3 + 4", "4 + 3", "(2 * 5) - 3"]
    assert unique_solution_count(solutions) == 2

def test_think_length_stats_no_think_block():
    completions = ["no think tags here"]
    stats = think_length_stats(completions)
    assert stats["mean"] == 0
```

---

### 12.16 `notebooks/00_data_exploration.ipynb` — **ADD ZERO-SHOT EVAL CELL**

The existing notebook likely just loads and previews the dataset. Add:

```python
# Cell: Zero-shot evaluation — run before any training
# If pass@1 > 0.65, switch to harder dataset or SmolLM2

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample 100 problems, run greedy decode, check correctness
# Log: pass@1, example outputs, length distribution
# Decision gate: print recommendation based on result
```

---

### Summary: File Change Priority

| Priority | File | Action | Reason |
|---|---|---|---|
| 🔴 Critical | `src/reward_functions.py` | Rewrite | Core of experiment; separability + degenerate filter missing |
| 🔴 Critical | `src/metrics.py` | Rewrite | pass@k, exploration gap, unique solution count all need correct implementation |
| 🔴 Critical | `tests/test_reward_functions.py` | Implement | Must pass before any training run |
| 🔴 Critical | `tests/test_metrics.py` | Implement | Must pass before any evaluation |
| 🟡 High | `configs/base.yaml` | Modify | ref_model, num_generations, log_reward_components must be correct |
| 🟡 High | `configs/c*.yaml` | Modify | β values, C5 must be 0.1 not 0.2, GSM8K refs removed |
| 🟡 High | `src/trainer.py` | Modify | ref_model load order, beta=0 handling, checkpoint resume |
| 🟡 High | `src/config.py` | Modify | Add hard_length_cap fields, remove GSM8K |
| 🟡 High | `src/dataset.py` | Modify | Remove GSM8K, verify field names, prompt format |
| 🟡 High | `src/prompts.py` | Modify | Verify format tags match parser, no reward leakage |
| 🟠 Medium | `src/logging_utils.py` | Modify | Separate component logging, CSV always-on |
| 🟠 Medium | `src/evaluation.py` | Modify | Temperature sampling for pass@k, LoRA load |
| 🟠 Medium | `scripts/train.py` | Modify | Config dump at start, checkpoint resume |
| 🟢 Low | `scripts/kaggle_setup.sh` | Modify | TRL version pin, sympy, wandb login |
| 🟢 Low | `notebooks/00_data_exploration.ipynb` | Add cell | Zero-shot eval gate |

---

## 13. Repository Structure

```
grpo-reward-hacking/
├── configs/
│   ├── base.yaml                # Shared hyperparameters [MODIFY]
│   ├── c1_baseline.yaml         [MODIFY]
│   ├── c2_hackable.yaml         [MODIFY]
│   ├── c3_kl_low.yaml           [MODIFY — β=0.01]
│   ├── c4_kl_med.yaml           [MODIFY — β=0.05]
│   ├── c5_kl_high.yaml          [MODIFY — β=0.1, NOT 0.2]
│   └── c6_length_cap.yaml       [MODIFY — add hard_length_cap fields]
│
├── src/
│   ├── config.py                [MODIFY — add RewardConfig fields]
│   ├── dataset.py               [MODIFY — remove GSM8K, verify field names]
│   ├── prompts.py               [MODIFY — verify format tags]
│   ├── reward_functions.py      [REWRITE — separable components + degenerate filter]
│   ├── metrics.py               [REWRITE — pass@k, gap, unique solutions]
│   ├── trainer.py               [MODIFY — ref_model load order, resume]
│   ├── evaluation.py            [MODIFY — temperature sampling, LoRA load]
│   ├── generation.py            [CHECK — likely fine as-is]
│   ├── logging_utils.py         [MODIFY — component logging, CSV always-on]
│   └── utils.py                 [CHECK — likely fine as-is]
│
├── scripts/
│   ├── train.py                 [MODIFY — config dump, resume]
│   ├── evaluate.py              [MODIFY — results save]
│   ├── run_all.sh               [CHECK — verify condition order]
│   └── kaggle_setup.sh          [MODIFY — version pins, wandb]
│
├── notebooks/
│   ├── 00_data_exploration.ipynb    [ADD zero-shot eval cell]
│   ├── 01_reward_analysis.ipynb     [CHECK]
│   └── 02_results_visualization.ipynb [CHECK]
│
├── docs/
│   └── trl_audit.md             [CREATE — record TRL source audit findings]
│
└── tests/
    ├── test_reward_functions.py  [IMPLEMENT — highest priority]
    ├── test_metrics.py           [IMPLEMENT — highest priority]
    └── ...                       [CHECK existing stubs]
```

---

## 14. Paper Structure

| Section | Content |
|---|---|
| **Introduction** | Reward hacking in RL for LLMs; exploration-exploitation tension; paper contribution |
| **Background** | GRPO mechanics; reward shaping; KL regularization in RLHF |
| **Experimental Setup** | Task, model, reward conditions, metrics, training details |
| **Results** | pass@k gap across conditions; length vs accuracy curves; rollout diversity |
| **Analysis** | Where guardrails help vs hurt; β sensitivity curve; C6 vs KL comparison |
| **Conclusion** | Practical guidance on guardrail tuning; limitations; open questions |

---

## 15. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| C2 doesn't hack (task too easy for Qwen) | Medium | Zero-shot eval first; fallback to harder problems or SmolLM2 |
| Kaggle session expires mid-run | High | Checkpoint every 100 steps; CSV backup; resume logic in trainer |
| β=0 doesn't fully disable KL in TRL | Medium | TRL source audit (12.0) before any training |
| KL and exploration collapse look the same in plots | Medium | Log components separately; length stats disambiguate |
| Single seed results not reproducible | High | 2 seeds minimum; report variance |
| pass@k gap too small to detect differences | Medium | Pilot run; if gap < 0.05 across all conditions, switch to harder task |
| Degenerate solutions inflate C2 pass@1 | High | Degenerate filter implemented and tested before any run |

---

## 16. References

- DeepSeekMath / DeepSeek-R1 tech report — original GRPO formulation
- HuggingFace TRL GRPOTrainer documentation and source
- `Jiayi-Pan/Countdown-Tasks-3to4` — HuggingFace dataset
- `open-thought/tiny-grpo` — minimal hackable GRPO reference implementation
- `policy-gradient/GRPO-Zero` — from-scratch GRPO, low GPU memory

---

*Last updated: July 2026 · Status: Design complete, implementation not started*
