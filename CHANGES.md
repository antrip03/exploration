# Changes — Codebase Audit and Optimization

Every file in `src/`, `scripts/`, `configs/`, plus `pyproject.toml` and
`requirements.txt`, was read before making any change. All 81 tests pass
(`python -m pytest tests/ -x -q`, 2 pre-existing skips unrelated to this work).

## Bug fixes

| # | File | Description |
|---|------|-------------|
| 1 | [src/logging_utils.py:102](src/logging_utils.py#L102) | `_setup_csv` opened the metrics CSV in `"w"` (write-only) mode, but `_extend_csv_header` later calls `.seek(0)` / `csv.DictReader(self._csv_file)` on it to rewrite the header when a new metric key appears mid-run — this raised `io.UnsupportedOperation: not readable`. Changed to `"w+"`. |
| 2 | [src/config.py:415-429](src/config.py#L415) | `to_yaml()` called `yaml.dump(self.to_dict(), ...)`, and `self.to_dict()` (`model_dump()`) leaves `Enum` fields (e.g. `RewardType`) as Python enum objects. PyYAML's default `Dumper` has no representer for arbitrary enum subclasses and falls back to unreadable `!!python/object/apply:...` tags. Changed to `self.model_dump(mode="json")`, which makes Pydantic v2 recursively coerce enums/etc. to plain JSON-safe values before the YAML dump. Verified: round-tripped config loads back with plain string values via `yaml.safe_load`. |
| 3 | [src/reward_functions.py](src/reward_functions.py) | Removed the `_DEBUG_PRINTS` global counter and the two associated `print(...)` blocks (one in `compute_reward_components`, one in `_make_reward_fn`) that dumped every completion's raw text and parse state to stdout for the first 5 calls of a training run. This is exactly the kind of ad-hoc debug output that shouldn't ship in the reward path (it runs on every rollout batch and pollutes training logs). No reward *computation* was touched — only print statements were removed. |
| 4 | `a.py` (root) | Deleted. It was a scratch script (`check_dataset.py`-style smoke test) duplicating what `scripts/zero_shot_eval.py` and the test suite already cover; not imported anywhere. |

### Claims from the task brief that did NOT need a fix

- **`scripts/modal_train.py` checkpoint path / `CHECKPOINT_REPO_MAP`** — the working tree already contains both fixes (`checkpoint_dir = .../f"{config_name}_s{seed}"/...` and a `CHECKPOINT_REPO_MAP` covering all seeds including 3 and 456). This was flagged as a bug during the earlier seed-3 investigation; the fix was already applied locally as an uncommitted edit before this session started. No further change made — flagging here so it isn't mistaken for unaddressed.
- **`src/evaluation.py` "missing" eval-seed fix** — `EvaluationPipeline.run()` already sets `random`/`numpy`/`torch`/`cuda` seeds from `eval_holdout_seed` before generation ([src/evaluation.py:46-54](src/evaluation.py#L46)). This is also already in place; no change made.
- **`RewardLoggingCallback` in `src/trainer.py`** — this class does not exist anywhere in the codebase (verified via full-repo search), so there was nothing to "fix." I did not fabricate a new callback class with no caller, since that would be dead code with no integration point. If a reward-logging callback is actually wanted, that's a new feature, not a bug fix — happy to build it against `ExperimentLogger.log_step` if useful.
- **"Double-wrapped model save" in `GRPOExperimentTrainer.save()`** — for the single-GPU LoRA path this repo actually runs, `self.trainer.model` is HF `Trainer`'s unwrapped model reference (Trainer keeps `self.model` unwrapped and uses a separate `self.model_wrapped` for the accelerator-wrapped copy), so it's already the plain `PeftModel` and `save_pretrained` works correctly today — this wasn't independently confirmed as an active bug. Left unchanged rather than adding an unverified PEFT-unwrap loop; flag if you've actually observed a wrapping issue on the multi-GPU path and I'll add the defensive unwrap.

## Performance optimizations (opt-in / non-invasive only)

No existing experimental hyperparameter (LR, beta, reward weights, LoRA rank/targets, batch sizes, seeds, prompts) was changed, per the constraint against making C1–C6 results incomparable.

| File | Change | Expected impact |
|------|--------|------------------|
| [src/trainer.py](src/trainer.py) `load_model()` | Enable `torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32` on CUDA | Free matmul speedup on Ampere (A10G) with no precision-relevant change for bf16 training |
| [src/trainer.py](src/trainer.py) `load_model()` | `AutoTokenizer.from_pretrained(..., use_fast=True)` | Rust-backed fast tokenizer (was previously implicit/default already for Qwen, now explicit) |
| [src/trainer.py](src/trainer.py) `load_model()` | Explicit warning when Flash Attention 2 isn't active (both fallback branches already logged, added a third summary warning) | Makes a silent 2-3x slowdown impossible to miss in logs |
| [src/trainer.py](src/trainer.py) `load_model()` | New opt-in `training.gradient_checkpointing` flag (default `false`) → `model.gradient_checkpointing_enable()` | Memory/compute tradeoff, useful if batch size needs to grow; off by default so it doesn't change current runs |
| [src/trainer.py](src/trainer.py) `load_model()` | New opt-in `training.compile_model` flag (default `false`) → `torch.compile(self.model)` | Potential 20-30% step-time speedup; off by default since `torch.compile` adds startup latency and interacts unpredictably with PEFT + generation-heavy GRPO rollouts — recommend testing on one condition before enabling broadly |
| [src/trainer.py](src/trainer.py) `setup_trainer()` | `dataloader_pin_memory=True` in `GRPOConfig` | Faster host→GPU transfer for the training dataloader |
| [src/generation.py](src/generation.py) `_generate_batch_once` | Pass `use_cache=True` explicitly to `model.generate(...)` | `load_model()` sets `model.config.use_cache=False` for training memory savings; without an explicit override this could leak into eval/generation calls that reuse the same model object and force slow non-cached autoregressive decoding |
| [configs/base.yaml](configs/base.yaml) | Added `gradient_checkpointing: false` and `compile_model: false` under `training`, and a comment on `lora.target_modules` documenting why it must not be narrowed for the existing conditions | Makes the new flags configurable per-condition without touching any current run |

Not implemented: LoRA target-module reduction (3.8) — left commented as instructed, not applied, since it would change trainable-parameter counts and invalidate comparisons across C1–C6. Modal image layer ordering (3.11) — already correct in the current `modal_train.py` (flash-attn wheel installs after the `pip_install` torch layer, and `add_local_dir` is the last image step).

## Structural cleanup

- Removed `a.py` (see above).
- `game24/` was already deleted from the working tree before this session (shows as `D` in `git status`); nothing further to do.
- Did **not** touch `agent.md`, `corrections.md`, `project_plan.md`, or `secrets.sh` — these are working notes / already-gitignored (`secrets.sh` is in `.gitignore`), not code, and deleting them risked destroying in-progress notes with no clear signal they're safe to remove. Flag explicitly if you want these gone.
- `outputs/` and `wandb/` are already in `.gitignore`.

## Requirements

`requirements.txt` core ML stack pinned to the exact Modal training image versions (previously `>=` floors that could drift):

```
torch==2.7.1
transformers==4.53.2
trl==0.19.1
peft==0.19.1
accelerate==1.14.0
datasets==5.0.0
sentencepiece==0.2.1
bitsandbytes==0.49.2
safetensors==0.8.0
pydantic==2.13.4
PyYAML==6.0.3
wandb==0.28.0
sentence-transformers==5.6.0
```
`tokenizers` pin removed (transitive pin via `transformers==4.53.2` is sufficient and avoids a version conflict). Everything else (scipy/pandas/sklearn/dev tools/notebook stack) left as floor-pinned since they aren't part of the reproducibility-critical training path.

## Verification

```bash
# CSV logger
python -c "from src.logging_utils import ExperimentLogger; print('OK')"

# Config serialization
python -c "
from src.config import ExperimentConfig
cfg = ExperimentConfig.from_yaml('configs/c1_baseline.yaml')
cfg.to_yaml('/tmp/test_config.yaml')
import yaml
with open('/tmp/test_config.yaml') as f:
    data = yaml.safe_load(f)
print('Config serialization OK:', list(data.keys()))
"

# Debug prints removed
grep -c "_DEBUG_PRINTS" src/reward_functions.py   # expect 0

# Full test suite
python -m pytest tests/ -x -q
```

All four ran clean; test suite: **81 passed, 2 skipped**.

## Not done — needs a GPU to verify

Section D of the brief (10-step timing baseline on A10G) requires an actual GPU; this environment is CPU-only (confirmed: `torch.cuda.is_available() == False`, no `nvidia-smi`). The `tf32`/pin-memory/gradient-checkpointing/compile changes are structurally correct and no-ops on CPU-only or when the corresponding config flag is off, but their real speedup should be measured on Modal before being treated as validated.
