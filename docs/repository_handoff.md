# Repository Handoff: GRPO Reward Hacking Experiments

This document is a planning-oriented handoff for the repository. It summarizes what is implemented, where the core logic lives, how the experiment workflow works, and what remains to be done for actual training runs and analysis.

## 1. Project Purpose

This repository implements a research workflow for studying reward hacking and exploration in GRPO for Countdown-style arithmetic reasoning tasks.

The experiments compare:
- a baseline reward signal,
- a hackable reward with a length bonus,
- KL-regularized variants,
- and a length-cap guardrail.

The central research question is whether guardrails reduce reward hacking while also reducing exploration diversity.

---

## 2. What Has Already Been Implemented

### Core experiment infrastructure
- Config-driven experiment setup with YAML configs for multiple conditions.
- Dataset loading for the Countdown task from Hugging Face.
- Prompt building and parsing for structured reasoning outputs.
- Reward function implementation with component-wise reward logging.
- GRPO training wrapper around TRL.
- Evaluation pipeline for greedy and sampled decoding.
- Metrics for pass@1, pass@k, exploration gap, unique solutions, and reasoning length.
- Logging and output management for training runs.
- Local and Kaggle-friendly setup scripts.

### Experiment conditions configured
The repository is prepared for the following conditions:
- C1: baseline
- C2: hackable
- C3: KL-low
- C4: KL-med
- C5: KL-high
- C6: length cap

### Verification and readiness
The repo has already been validated locally:
- unit tests pass,
- the zero-shot evaluation entrypoint runs,
- the training/evaluation scripts are wired up,
- and the environment setup is documented for both local and Kaggle use.

---

## 3. Repository Structure

### Top level
- [README.md](../README.md): high-level overview and quick-start commands.
- [pyproject.toml](../pyproject.toml): project metadata and pytest configuration.
- [requirements.txt](../requirements.txt): Python dependencies.
- [project_plan.md](../project_plan.md): original research plan and experimental design.
- [LICENSE](../LICENSE): project license.

### Configs
- [configs/base.yaml](../configs/base.yaml): shared defaults for all conditions.
- [configs/c1_baseline.yaml](../configs/c1_baseline.yaml): baseline condition.
- [configs/c2_hackable.yaml](../configs/c2_hackable.yaml): hackable condition.
- [configs/c3_kl_low.yaml](../configs/c3_kl_low.yaml): KL-regularized low beta.
- [configs/c4_kl_med.yaml](../configs/c4_kl_med.yaml): KL-regularized medium beta.
- [configs/c5_kl_high.yaml](../configs/c5_kl_high.yaml): KL-regularized high beta.
- [configs/c6_length_cap.yaml](../configs/c6_length_cap.yaml): hard length-cap condition.

### Source code
- [src/config.py](../src/config.py): Pydantic-based configuration models.
- [src/dataset.py](../src/dataset.py): dataset loading and prompt preprocessing.
- [src/prompts.py](../src/prompts.py): prompt templates and output parsing helpers.
- [src/reward_functions.py](../src/reward_functions.py): reward computation and component breakdown.
- [src/trainer.py](../src/trainer.py): trainer wrapper around TRL GRPO.
- [src/generation.py](../src/generation.py): generation helpers for greedy and sampled decoding.
- [src/evaluation.py](../src/evaluation.py): evaluation pipeline.
- [src/metrics.py](../src/metrics.py): pass@1/pass@k/exploration metrics.
- [src/logging_utils.py](../src/logging_utils.py): logging setup.
- [src/utils.py](../src/utils.py): seed handling, checkpoint utilities, GPU detection, and file helpers.

### Scripts
- [scripts/train.py](../scripts/train.py): start a single experiment run.
- [scripts/evaluate.py](../scripts/evaluate.py): evaluate a checkpoint.
- [scripts/zero_shot_eval.py](../scripts/zero_shot_eval.py): run a quick zero-shot gate before training.
- [scripts/run_all.py](../scripts/run_all.py): cross-platform launcher for multiple conditions.
- [scripts/run_all.sh](../scripts/run_all.sh): bash launcher for the same workflow.
- [scripts/kaggle_setup.sh](../scripts/kaggle_setup.sh): Kaggle environment setup script.

### Tests
- [tests](../tests): unit tests for config parsing, data loading, prompts, rewards, metrics, evaluation, generation, logging, and trainer behavior.

### Docs
- [docs/installation.md](../docs/installation.md): installation instructions.
- [docs/trl_audit.md](../docs/trl_audit.md): notes on TRL behavior and assumptions.
- [docs/repository_overview.md](../docs/repository_overview.md): repository overview.
- [docs/repository_handoff.md](../docs/repository_handoff.md): this document.

---

## 4. What Each Main File Does

### [src/config.py](../src/config.py)
Defines all configuration dataclasses and Pydantic models.

This is the central place where experiment settings are validated and loaded from YAML. It covers:
- model config,
- LoRA config,
- training hyperparameters,
- reward settings,
- generation settings,
- logging settings,
- dataset settings.

If you want to add a new hyperparameter or condition, this is usually the first place to edit.

### [src/dataset.py](../src/dataset.py)
Loads the Countdown dataset and converts it into the prompt/answer schema expected by training and evaluation.

Key responsibilities:
- load the Hugging Face dataset,
- split into train/eval subsets,
- convert raw examples into prompt/answer/target/nums fields,
- support optional evaluation holdout creation when no eval split is present.

### [src/prompts.py](../src/prompts.py)
Builds prompts and parses model outputs.

This file defines:
- the Countdown prompt template,
- system prompt wording,
- chat message formatting,
- and extraction of the answer and reasoning blocks from model output.

### [src/reward_functions.py](../src/reward_functions.py)
Implements the reward logic used by GRPO.

Important features:
- correctness reward,
- format bonus,
- length bonus,
- hard length-cap handling,
- degenerate solution filtering,
- and component-wise logging for correctness/length/format.

This is the most important file for the actual experiment behavior.

### [src/trainer.py](../src/trainer.py)
Wraps TRL GRPO training.

Responsibilities:
- load the base model and tokenizer,
- configure LoRA,
- build the GRPO trainer,
- run training,
- save checkpoints.

### [src/generation.py](../src/generation.py)
Provides utilities for generating completions.

Used for:
- greedy generation,
- batched generation,
- and sampling k completions per prompt.

### [src/evaluation.py](../src/evaluation.py)
Runs evaluation on a checkpoint.

It loads a saved checkpoint, generates greedy and sampled completions, computes metrics, and writes evaluation artifacts.

### [src/metrics.py](../src/metrics.py)
Computes research metrics.

Includes:
- pass@1,
- pass@k,
- exploration gap,
- unique solution count,
- reasoning length statistics.

### [src/logging_utils.py](../src/logging_utils.py)
Configures experiment logging and output directories.

### [src/utils.py](../src/utils.py)
Contains utility helpers for:
- random seeding,
- GPU detection,
- checkpoint list/find helpers,
- JSON/JSONL saving,
- Kaggle-oriented helpers.

---

## 5. How the Training Workflow Works

### Local or Kaggle entrypoints
1. Load a YAML config.
2. Build the dataset.
3. Initialize the trainer wrapper.
4. Load the base model and tokenizer.
5. Configure LoRA and GRPO settings.
6. Train with the selected reward function.
7. Save checkpoints and logs.
8. Evaluate the saved checkpoint.

### Typical command flow
- Zero-shot gate:
  - `python scripts/zero_shot_eval.py --config configs/c1_baseline.yaml`
- Single training run:
  - `python scripts/train.py --config configs/c2_hackable.yaml`
- Evaluation:
  - `python scripts/evaluate.py --config configs/c2_hackable.yaml --checkpoint outputs/c2_hackable/checkpoint-final`
- Multi-condition sweep:
  - `python scripts/run_all.py`

---

## 6. How the Reward Setup Works

The reward logic is designed to make reward hacking measurable.

### Baseline condition
- correctness only,
- format bonus.

### Hackable condition
- correctness,
- length bonus,
- format bonus.

### Guardrail conditions
- same as hackable, but with KL regularization or a hard length cap.

### Important implementation details
- length and format rewards are logged separately,
- degenerate solutions are filtered out,
- the reward function is designed to support post-hoc analysis of hacking behavior.

---

## 7. What Is Already Done vs What Still Needs Work

### Done
- repository scaffold,
- experiment config system,
- reward functions,
- dataset pipeline,
- training/evaluation entrypoints,
- tests,
- Kaggle setup scripts,
- local validation.

### Still to do for the research workflow
- run pilot experiments,
- calibrate reward weights,
- calibrate the hard length cap,
- inspect whether the hackable condition actually exhibits hacking behavior,
- run the main conditions across seeds,
- collect metrics and produce plots,
- write the paper or report.

---

## 8. Recommended Next Steps

### Phase A: Pilot validation
1. Run the zero-shot gate.
2. Run a short C2 pilot.
3. Inspect whether the model lengthens its reasoning and whether accuracy remains flat or declines.
4. Calibrate reward weights and the length cap.

### Phase B: Main experimental sweep
1. Run the baseline condition.
2. Run the hackable condition.
3. Run KL-regularized variants.
4. Run the hard length-cap condition.
5. Repeat with a second seed if feasible.

### Phase C: Analysis
1. Compare pass@1 and pass@k across conditions.
2. Plot exploration gap over time.
3. Analyze reasoning length distributions.
4. Compare reward component breakdowns.
5. Summarize findings for the paper.

---

## 9. Notes for a Planning LLM

When planning follow-up work, treat this repository as an experiment harness rather than a generic ML app. The main deliverables are not product features but research runs and analyses.

The most important files to focus on for planning are:
- [src/reward_functions.py](../src/reward_functions.py) for reward behavior,
- [src/trainer.py](../src/trainer.py) for training setup,
- [src/metrics.py](../src/metrics.py) for evaluation framing,
- [configs](../configs) for experiment variants,
- [scripts/train.py](../scripts/train.py) and [scripts/evaluate.py](../scripts/evaluate.py) for execution.

If a planning agent needs to decide what to do next, the best high-level sequence is:
1. validate the pilot,
2. calibrate reward settings,
3. run the main conditions,
4. analyze the outputs,
5. write up the results.

---

## 10. Practical Guidance

- Use the configs rather than editing code for most experimental changes.
- Keep changes small and experiment-focused.
- Save outputs per condition under the corresponding output directory.
- Prefer a short pilot before any long training run.
- Treat the reward function as the primary lever for the research question.

This repository is now in a usable state for running and analyzing GRPO experiments, especially for the Countdown task on Kaggle or local GPU hardware.
