# Experiment Roadmap

This document tracks the planned research phases for the paper:
**"Do Reward Hacking Guardrails Reduce Exploration Diversity in GRPO?"**

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🟡 | In progress |
| ✅ | Complete |
| ❌ | Blocked |

---

## Phase 0 — Repository Setup ✅

- ✅ Project structure and skeleton code
- ✅ Configuration system (Pydantic configs + YAML)
- ✅ Prompt templates and output parsers
- ✅ Reward function interfaces
- ✅ Metric function stubs
- ✅ Trainer wrapper skeleton
- ✅ Logging infrastructure
- ✅ Test suite skeletons
- ✅ Documentation

---

## Phase 1 — Data & Reward Implementation ⬜

### 1.1 Dataset Loading
- ⬜ Implement `_load_countdown()` — decide: HF Hub vs. synthetic
  - Candidate: `Jiayi-Pan/Countdown-Tasks-3to4`
- ⬜ Implement `_load_gsm8k()` — load from `"gsm8k"` HF dataset
- ⬜ Implement `preprocess_countdown_example()` field mapping
- ⬜ Implement `preprocess_gsm8k_example()` answer extraction (#### split)
- ⬜ Add train/eval split subsetting (max_train_samples, max_eval_samples)

### 1.2 Reward Functions
- ⬜ Implement `answer_reward()` — numeric equivalence checking
  - Countdown: evaluate arithmetic expression string
  - GSM8K: parse and compare numeric answer
- ⬜ Implement `format_reward()` — graduated tag presence scoring
- ⬜ Implement `length_bonus()` — tokenizer-based token count
- ⬜ Implement `length_penalty()` — tokenizer-based cap enforcement
- ⬜ Decide KL penalty approach: TRL's built-in `kl_coef` vs. manual computation

### 1.3 Metrics
- ⬜ Implement `pass_at_1()` — wire up `answer_reward()`
- ⬜ Implement `pass_at_k()` — Chen et al. (2021) unbiased estimator
- ⬜ Implement `unique_solution_count()` — distinct correct expressions
- ⬜ Implement `embedding_variance()` — SentenceTransformer embeddings
- ⬜ Validate all metrics on toy data

**Milestone:** All reward functions and metrics return correct values on dummy data.

---

## Phase 2 — Model & Trainer Implementation ⬜

### 2.1 Model Loading
- ⬜ Implement `load_model()` — AutoModelForCausalLM + LoRA
- ⬜ Implement `_build_lora_config()` — PEFT LoraConfig
- ⬜ Verify FA2 fallback for Kaggle (T4 GPU)
- ⬜ Log trainable parameter count

### 2.2 Trainer Setup
- ⬜ Implement `setup_trainer()` — TRL GRPOConfig + GRPOTrainer
- ⬜ Wire reward function to TRL's `reward_funcs`
- ⬜ Map cfg fields to GRPOConfig fields exactly

### 2.3 Generation
- ⬜ Implement `generate_single()` — single prompt, single completion
- ⬜ Implement `generate_batch()` — batched with padding
- ⬜ Implement `generate_k_completions()` — k samples per prompt
- ⬜ Add OOM handling (dynamic batch size reduction)

### 2.4 Training Loop
- ⬜ Implement `train()` — `self.trainer.train()`
- ⬜ Add keyboard interrupt handler (save on Ctrl+C)
- ⬜ Implement `evaluate()` — compute MetricsResult during training

**Milestone:** C1 (baseline) trains end-to-end on a toy dataset without errors.

---

## Phase 3 — Logging & Evaluation ⬜

### 3.1 Logging Backends
- ⬜ Implement `_setup_wandb()` — `wandb.init()`
- ⬜ Implement `_setup_tensorboard()` — SummaryWriter
- ⬜ Implement `_setup_csv()` — CSV file + header
- ⬜ Implement `log()` — write to all active backends
- ⬜ Implement `log_config()` — upload config to W&B

### 3.2 Evaluation Pipeline
- ⬜ Implement `EvaluationPipeline.run()`
- ⬜ Implement `save_results()` — metrics.json, completions.jsonl
- ⬜ Implement `compare_conditions()` — cross-condition DataFrame

### 3.3 CLI Scripts
- ⬜ Wire up `scripts/train.py` (remove all `# TODO:` comments)
- ⬜ Wire up `scripts/evaluate.py`
- ⬜ Test `run_all.sh` dry run

**Milestone:** All 6 configs can be run via `python scripts/train.py --config configs/cX_...yaml`

---

## Phase 4 — Experiments ⬜

> **Do not start Phase 4 until Phases 1–3 are verified on toy data.**

### 4.1 Pilot Run
- ⬜ C1 baseline — 50 steps on small data subset
- ⬜ Verify reward signal is non-trivial (not 0 or 1 throughout)
- ⬜ Verify logging to W&B and CSV

### 4.2 Full Runs (Countdown)
- ⬜ C1 — Baseline (control)
- ⬜ C2 — Hackable (no guardrail)
- ⬜ C3 — Hackable + KL β=0.01
- ⬜ C4 — Hackable + KL β=0.05
- ⬜ C5 — Hackable + KL β=0.1
- ⬜ C6 — Hackable + length cap

**Seed sweep:** Run each condition with seeds {42, 123, 456} for error bars.

### 4.3 Secondary Task (GSM8K)
- ⬜ Repeat C1, C2, C4, C6 on GSM8K to test generalisability

### 4.4 Analysis
- ⬜ Plot pass@1 and pass@k curves across training steps
- ⬜ Plot exploration_gap vs. training step per condition
- ⬜ Plot embedding_variance across conditions
- ⬜ Plot reasoning_length distributions (box plots)
- ⬜ Statistical significance testing (t-test across seeds)

---

## Phase 5 — Paper ⬜

- ⬜ Write results section with figures from Phase 4 analysis
- ⬜ Ablation: beta sweep figures (C3 vs. C4 vs. C5)
- ⬜ Ablation: KL vs. length cap comparison (C4 vs. C6)
- ⬜ Limitations section
- ⬜ Final arXiv submission

---

## Open Research Questions

1. **KL source** — Should we use TRL's built-in KL (`kl_coef` in GRPOConfig) or implement a manual KL penalty in the reward function? The former is simpler; the latter gives more control.

2. **Countdown dataset** — Should we use `Jiayi-Pan/Countdown-Tasks-3to4` from HF Hub or generate examples synthetically? The synthetic approach is fully controllable but requires implementing a valid target-generation algorithm.

3. **Exploration proxy** — Is `embedding_variance` of think blocks a reliable proxy for exploration diversity? Should we also measure action diversity (unique final answers) separately from reasoning diversity?

4. **Length bonus schedule** — Should `length_bonus_weight` be annealed during training to prevent full exploitation, or kept constant throughout?

5. **Evaluation checkpoint** — Should we evaluate at the final checkpoint only, or track all metrics through training (which is more informative but expensive)?

---

## Timeline (Estimated)

| Phase | Duration | Target |
|-------|----------|--------|
| 0 — Setup | Done | Week 1 |
| 1 — Data & Reward | 1 week | Week 2 |
| 2 — Model & Trainer | 1 week | Week 3 |
| 3 — Logging & Eval | 3 days | Week 3 |
| 4 — Experiments | 1–2 weeks | Week 5 |
| 5 — Paper | 1 week | Week 6 |
