"""
src/config.py
=============
Configuration dataclasses / Pydantic models for all experiment components.

All configuration is defined here and loaded from YAML files at runtime.
Using Pydantic v2 for validation and type safety.

Usage:
    from src.config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("configs/c1_baseline.yaml")
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class RewardType(str, Enum):
    """Supported reward function types."""

    BASELINE = "baseline"
    HACKABLE = "hackable"
    GUARDRAILED = "guardrailed"


class DatasetName(str, Enum):
    """Supported dataset names."""

    COUNTDOWN = "countdown"


class LogLevel(str, Enum):
    """Logging verbosity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Configuration for the base language model.

    Attributes:
        name: HuggingFace model identifier.
        ref_model: HuggingFace reference model identifier.
        revision: Git revision of the model (tag / branch / commit SHA).
        dtype: Torch dtype string ('bfloat16', 'float16', 'float32').
        attn_implementation: Attention backend ('flash_attention_2' or 'eager').
        trust_remote_code: Allow loading custom model code from HF Hub.
        max_length: Maximum sequence length (prompt + generation).
    """

    name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model name or local path.",
    )
    ref_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace reference model name or local path.",
    )
    revision: str = Field(default="main", description="Model revision / tag.")
    dtype: str = Field(default="bfloat16", description="Torch computation dtype.")
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation backend.",
    )
    trust_remote_code: bool = Field(default=True)
    max_length: int = Field(default=2048, ge=1)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Ensure dtype is one of the supported Torch dtypes."""
        allowed = {"bfloat16", "float16", "float32", "float64"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning via PEFT.

    Attributes:
        enabled: Whether to use LoRA.
        r: Rank of the low-rank decomposition matrices.
        alpha: LoRA scaling factor (effective lr scale = alpha / r).
        lora_alpha: Alternate name for alpha.
        dropout: Dropout rate applied to LoRA layers.
        lora_dropout: Alternate name for dropout.
        target_modules: List of module names to apply LoRA to.
        bias: Which bias parameters to train ('none', 'all', 'lora_only').
        task_type: PEFT task type string.
    """

    enabled: bool = Field(default=True)
    r: int = Field(default=16, ge=1, description="LoRA rank.")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling factor.")
    lora_alpha: Optional[int] = Field(default=None)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_dropout: Optional[float] = Field(default=None)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")

    @model_validator(mode="after")
    def sync_lora_fields(self) -> "LoRAConfig":
        if self.lora_alpha is not None:
            self.alpha = self.lora_alpha
        else:
            self.lora_alpha = self.alpha
        if self.lora_dropout is not None:
            self.dropout = self.lora_dropout
        else:
            self.lora_dropout = self.dropout
        return self


class TrainingConfig(BaseModel):
    """Configuration for GRPOTrainer training loop.

    Attributes:
        output_dir: Directory to save checkpoints and logs.
        max_steps: Total number of training steps.
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps before optimizer update.
        learning_rate: Peak learning rate.
        lr_scheduler_type: LR scheduler name.
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: AdamW weight decay coefficient.
        max_grad_norm: Gradient clipping threshold.
        seed: Global random seed for reproducibility.
        num_generations: Number of rollout samples per prompt (GRPO G).
        temperature: Sampling temperature for rollouts.
        top_p: Nucleus sampling threshold.
        save_steps: Checkpoint save frequency.
        eval_steps: Evaluation frequency.
        logging_steps: Logging frequency.
        bf16: Whether to use bfloat16 mixed precision.
        fp16: Whether to use float16 mixed precision (mutually exclusive with bf16).
        beta: KL divergence penalty coefficient (0 = disabled).
        max_completion_length: Maximum completion length.
    """

    output_dir: str = Field(default="outputs/experiment")
    max_steps: int = Field(default=500, ge=1)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    lr_scheduler_type: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=42)
    dataloader_num_workers: int = Field(default=2, ge=0)
    remove_unused_columns: bool = Field(default=False)
    # GRPO-specific
    num_generations: int = Field(default=8, ge=2, description="G — rollouts per prompt.")
    generation_batch_size: Optional[int] = Field(default=8, ge=2)
    temperature: float = Field(default=0.9, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    beta: float = Field(default=0.0, ge=0.0, description="KL coefficient; 0.0 = no KL")
    max_completion_length: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def check_precision_flags(self) -> "TrainingConfig":
        """Ensure fp16 and bf16 are not both enabled."""
        if self.fp16 and self.bf16:
            raise ValueError("Only one of fp16 or bf16 can be enabled at a time.")
        if (
            self.generation_batch_size is not None
            and self.generation_batch_size % self.num_generations != 0
        ):
            raise ValueError("generation_batch_size must be divisible by num_generations")
        return self


class RewardConfig(BaseModel):
    """Configuration for the reward function.

    Attributes:
        type: Reward function type (baseline | hackable | guardrailed).
        correctness_weight: Weight for answer-correctness reward.
        length_bonus_max: Maximum weight for length bonus.
        length_bonus_ceiling: Token ceiling for length bonus.
        format_bonus: Weight for correct <think>/<answer> format.
        hard_length_cap: Use hard reasoning length cap.
        hard_length_cap_tokens: Token count threshold for hard length cap.
        answer_reward_weight: Legacy compatibility key.
        format_reward_weight: Legacy compatibility key.
        length_bonus_weight: Legacy compatibility key.
        kl_beta: Legacy compatibility key.
        max_reasoning_tokens: Legacy compatibility key.
    """

    type: RewardType = Field(default=RewardType.BASELINE)
    correctness_weight: float = Field(default=1.0, ge=0.0)
    length_bonus_max: float = Field(default=0.0, ge=0.0)
    length_bonus_ceiling: int = Field(default=512, ge=1)
    format_bonus: float = Field(default=0.0, ge=0.0)
    hard_length_cap: bool = Field(default=False)
    hard_length_cap_tokens: int = Field(default=0, ge=0)

    # Legacy variables for compatibility with existing configs
    answer_reward_weight: float = Field(default=1.0, ge=0.0)
    format_reward_weight: float = Field(default=0.0, ge=0.0)
    length_bonus_weight: float = Field(default=0.0, ge=0.0)
    kl_beta: float = Field(default=0.0, ge=0.0)
    max_reasoning_tokens: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> "RewardConfig":
        # If legacy fields are set, map them to standard ones
        if self.answer_reward_weight != 1.0 and self.correctness_weight == 1.0:
            self.correctness_weight = self.answer_reward_weight
        if self.format_reward_weight != 0.0 and self.format_bonus == 0.0:
            self.format_bonus = self.format_reward_weight
        if self.length_bonus_weight != 0.0 and self.length_bonus_max == 0.0:
            self.length_bonus_max = self.length_bonus_weight
        if self.max_reasoning_tokens is not None:
            self.hard_length_cap = True
            self.hard_length_cap_tokens = self.max_reasoning_tokens
        return self


class GenerationConfig(BaseModel):
    """Configuration for text generation (evaluation / inference).

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: Whether to use sampling (True) or greedy (False).
        temperature: Sampling temperature.
        top_p: Nucleus sampling p.
        top_k: Top-k sampling k.
        repetition_penalty: Penalty for repeated n-grams.
        num_return_sequences: Number of sequences to generate per prompt.
    """

    max_new_tokens: int = Field(default=512, ge=1)
    do_sample: bool = Field(default=True)
    temperature: float = Field(default=0.9, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    num_return_sequences: int = Field(default=1, ge=1)


class LoggingConfig(BaseModel):
    """Configuration for experiment logging backends.

    Attributes:
        level: Python logging level.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (user or team). Reads WANDB_ENTITY env var if None.
        use_tensorboard: Enable TensorBoard logging.
        use_csv: Enable CSV metrics logging.
        log_dir: Directory for all log files.
        log_reward_components: Whether to log correctness/length/format separately.
    """

    level: LogLevel = Field(default=LogLevel.INFO)
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="grpo-reward-hacking")
    wandb_entity: Optional[str] = Field(default=None)
    use_tensorboard: bool = Field(default=True)
    use_csv: bool = Field(default=True)
    log_dir: str = Field(default="outputs/logs")
    log_reward_components: bool = Field(default=True)


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing.

    Attributes:
        name: Dataset identifier.
        split_train: Training split name.
        split_eval: Evaluation split name.
        max_train_samples: Subset size for training (None = all).
        max_eval_samples: Subset size for evaluation (None = all).
        preprocessing_num_workers: Workers for dataset.map().
        countdown_min_digits: Minimum digit count per number (Countdown).
        countdown_max_digits: Maximum digit count per number (Countdown).
        countdown_num_numbers: Numbers in each Countdown puzzle.
    """

    name: str = Field(default="Jiayi-Pan/Countdown-Tasks-3to4")
    split_train: str = Field(default="train")
    split_eval: Optional[str] = Field(
        default=None,
        description="Evaluation split. If absent, reserve a deterministic holdout from train.",
    )
    eval_holdout_seed: int = Field(default=42)
    max_train_samples: Optional[int] = Field(default=None, ge=1)
    max_eval_samples: Optional[int] = Field(default=200, ge=1)
    preprocessing_num_workers: int = Field(default=4, ge=1)
    # Countdown-specific
    countdown_min_digits: int = Field(default=1, ge=1)
    countdown_max_digits: int = Field(default=4, ge=1)
    countdown_num_numbers: int = Field(default=6, ge=2)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level ExperimentConfig
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentConfig(BaseModel):
    """Top-level configuration for a full experiment run.

    Composes all sub-configs. Loaded from a YAML file.

    Attributes:
        condition_id: Unique identifier for this experimental condition.
        description: Human-readable description.
        tags: Arbitrary tags for filtering / grouping in W&B.
        model: Model configuration.
        lora: LoRA configuration.
        training: Training loop configuration.
        reward: Reward function configuration.
        generation: Generation / inference configuration.
        logging: Logging backend configuration.
        dataset: Dataset loading configuration.
    """

    condition_id: str = Field(default="experiment")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an ExperimentConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully validated ExperimentConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValidationError: If the YAML contents fail Pydantic validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        defaults = raw.pop("defaults", [])
        if defaults:
            merged: dict[str, Any] = {}
            for default in defaults:
                default_name = default if isinstance(default, str) else next(iter(default))
                default_path = path.parent / f"{default_name}.yaml"
                if not default_path.exists():
                    raise FileNotFoundError(f"Base config not found: {default_path}")
                with open(default_path, "r", encoding="utf-8") as f:
                    base_raw = yaml.safe_load(f) or {}
                base_raw.pop("defaults", None)
                merged = _deep_merge(merged, base_raw)
            raw = _deep_merge(merged, raw)
        # Flatten nested 'experiment' key if present
        experiment_meta = raw.pop("experiment", {})
        raw.update(experiment_meta)
        raw = _resolve_placeholders(raw)
        logger.info("Loaded config from %s (condition_id=%s)", path, raw.get("condition_id"))
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain dictionary.

        Returns:
            A recursively serialized dictionary representation.
        """
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        """Serialize and save this config to a YAML file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config to %s", path)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_placeholders(value: Any, condition_id: Optional[str] = None) -> Any:
    """Resolve the project-specific condition placeholder in strings."""
    if isinstance(value, dict):
        condition = value.get("condition_id", condition_id)
        return {key: _resolve_placeholders(item, condition) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, condition_id) for item in value]
    if isinstance(value, str) and condition_id is not None:
        return value.replace("${experiment.condition_id}", condition_id)
    return value
