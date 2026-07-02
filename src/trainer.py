"""Thin, version-pinned wrapper around TRL's GRPOTrainer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from src.config import ExperimentConfig
from src.reward_functions import get_reward_fn
from src.utils import CheckpointManager

logger = logging.getLogger(__name__)

try:
    import transformers  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight local test envs
    transformers = ModuleType("transformers")

    class _MissingAutoTokenizer:  # type: ignore[no-redef]
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError("transformers is not installed; install requirements.txt first")

    class _MissingAutoModelForCausalLM:  # type: ignore[no-redef]
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError("transformers is not installed; install requirements.txt first")

    transformers.AutoTokenizer = _MissingAutoTokenizer
    transformers.AutoModelForCausalLM = _MissingAutoModelForCausalLM
    sys.modules.setdefault("transformers", transformers)


class GRPOExperimentTrainer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self._reward_fn = get_reward_fn(cfg.reward)
        self._runtime_bf16 = cfg.training.bf16
        self._runtime_fp16 = cfg.training.fp16

    def _torch_dtype(self) -> Any:
        import torch

        requested = getattr(torch, self.cfg.model.dtype)
        if requested is torch.bfloat16 and torch.cuda.is_available():
            supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if not supported:
                logger.warning("GPU does not support bfloat16; using float16")
                self._runtime_bf16, self._runtime_fp16 = False, True
                return torch.float16
        return requested

    def load_model(self) -> None:
        """Load the base model only; TRL applies PEFT after preserving its reference."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.cfg.model.ref_model != self.cfg.model.name:
            raise ValueError(
                "This study requires ref_model to equal the untouched base model; "
                f"got {self.cfg.model.ref_model!r} and {self.cfg.model.name!r}"
            )
        common = {
            "revision": self.cfg.model.revision,
            "trust_remote_code": self.cfg.model.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name, **common)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            **common,
            "torch_dtype": self._torch_dtype(),
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.name,
                attn_implementation=self.cfg.model.attn_implementation,
                **model_kwargs,
            )
        except (ImportError, ValueError) as exc:
            if self.cfg.model.attn_implementation != "flash_attention_2":
                raise
            logger.warning("FlashAttention 2 unavailable (%s); falling back to eager", exc)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.name,
                attn_implementation="eager",
                **model_kwargs,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = False
        self._reward_fn = get_reward_fn(self.cfg.reward, tokenizer=self.tokenizer)
        logger.info("Model loaded on %s", device)
        self.log_model_info()

    def _build_lora_config(self) -> Any:
        from peft import LoraConfig

        return LoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
            bias=self.cfg.lora.bias,
            task_type=self.cfg.lora.task_type,
        )

    def setup_trainer(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> None:
        """Build TRL 0.19's trainer with built-in KL and PEFT reference semantics."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_model() before setup_trainer().")
        from trl import GRPOConfig, GRPOTrainer

        t = self.cfg.training
        report_to = []
        if self.cfg.logging.use_wandb:
            report_to.append("wandb")
        if self.cfg.logging.use_tensorboard:
            report_to.append("tensorboard")
        args = GRPOConfig(
            output_dir=t.output_dir,
            max_steps=t.max_steps,
            per_device_train_batch_size=t.per_device_train_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            generation_batch_size=t.generation_batch_size,
            learning_rate=t.learning_rate,
            lr_scheduler_type=t.lr_scheduler_type,
            warmup_ratio=t.warmup_ratio,
            weight_decay=t.weight_decay,
            max_grad_norm=t.max_grad_norm,
            seed=t.seed,
            dataloader_num_workers=t.dataloader_num_workers,
            remove_unused_columns=False,
            num_generations=t.num_generations,
            temperature=t.temperature,
            top_p=t.top_p,
            top_k=self.cfg.generation.top_k,
            max_completion_length=t.max_completion_length or self.cfg.generation.max_new_tokens,
            beta=t.beta,
            save_strategy="steps",
            save_steps=t.save_steps,
            save_total_limit=3,
            logging_steps=t.logging_steps,
            eval_strategy="no",
            bf16=self._runtime_bf16,
            fp16=self._runtime_fp16,
            report_to=report_to,
            run_name=self.cfg.condition_id,
            log_completions=True,
        )
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[self._reward_fn],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            peft_config=self._build_lora_config() if self.cfg.lora.enabled else None,
        )

    def train(self, resume: bool = True) -> Any:
        """Train, automatically resuming the most recent valid checkpoint."""
        if self.trainer is None:
            raise RuntimeError("Call setup_trainer() before train().")
        checkpoint = (
            CheckpointManager(self.cfg.training.output_dir).latest_checkpoint()
            if resume
            else None
        )
        logger.info("Training %s (resume=%s)", self.cfg.condition_id, checkpoint)
        result = self.trainer.train(
            resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None
        )
        self.save()
        return result

    def evaluate(self, eval_dataset: Optional[Any] = None) -> dict[str, float]:
        if self.trainer is None:
            raise RuntimeError("Call setup_trainer() before evaluate().")
        return self.trainer.evaluate(eval_dataset=eval_dataset)

    def save(self, path: Optional[str] = None) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model is loaded")
        destination = Path(path or self.cfg.training.output_dir) / "checkpoint-final"
        destination.mkdir(parents=True, exist_ok=True)
        model = self.trainer.model if self.trainer is not None else self.model
        model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        self.cfg.to_yaml(destination / "experiment_config.yaml")

    def load(self, path: str) -> None:
        """Load a saved LoRA adapter for inference."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=self._torch_dtype(),
            trust_remote_code=self.cfg.model.trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if (checkpoint / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(base, checkpoint)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=self._torch_dtype(),
                device_map="auto",
            )
        self.model.eval()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.trainer is not None

    def log_model_info(self) -> None:
        if self.model is None:
            return
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info(
            "Base model parameters before PEFT: %d trainable / %d total",
            trainable,
            total,
        )
