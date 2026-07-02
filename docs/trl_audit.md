# TRL GRPO Audit

Source checked against TRL 0.19.0 documentation and trainer source.

## Findings

- `GRPOTrainer` accepts `model`, `reward_funcs`, `args`, `train_dataset`, `eval_dataset`, `processing_class`, and optional `peft_config`.
- `beta=0` disables the KL branch and the trainer does not need a separate reference model in that case.
- With PEFT enabled, TRL’s reference semantics rely on the frozen base model behavior rather than a separately trained LoRA reference.
- Custom reward functions receive `completions` plus dataset columns such as `prompts`, `answers`, `target`, and `nums` when those columns are present.
- `num_generations` is the number of samples per prompt used for GRPO rollouts.
- `generation_batch_size` must be divisible by `num_generations`.

## Configuration impact

- Keep `beta: 0.0` for C1, C2, and C6.
- Use `generation_batch_size: 8` with `num_generations: 8`.
- Load the same base checkpoint for `model` and `ref_model` in the config, but let TRL handle the reference path according to `beta`.
