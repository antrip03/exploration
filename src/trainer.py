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

    def _ensure_flash_attention(self) -> None:
        if self.cfg.model.attn_implementation != "flash_attention_2":
            return
        try:
            import importlib.util
            if importlib.util.find_spec("flash_attn") is not None:
                logger.info("FlashAttention 2 package already available")
                return
        except Exception:
            pass

        logger.info("Attempting to install flash-attn for FlashAttention 2")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ninja", "packaging"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"])
            logger.info("flash-attn installed successfully")
        except Exception as exc:  # pragma: no cover - environment-dependent
            logger.warning("Could not install flash-attn; falling back to eager attention (%s)", exc)

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

        attn_impl = self.cfg.model.attn_implementation
        if attn_impl == "flash_attention_2":
            try:
                import importlib.util
                if importlib.util.find_spec("flash_attn") is None:
                    raise ImportError("flash_attn not installed")
            except Exception as exc:
                logger.warning("FlashAttention 2 unavailable (%s); forcing eager attention", exc)
                attn_impl = "eager"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.name,
                attn_implementation=attn_impl,
                **model_kwargs,
            )
        except (ImportError, ValueError) as exc:
            logger.warning("Attention backend %r unavailable (%s); falling back to eager", attn_impl, exc)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.name,
                attn_implementation="eager",
                **model_kwargs,
            )

        # For multi-GPU (accelerate), let TRL handle device placement; otherwise place on single device
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                logger.info("Multi-GPU detected; deferring device placement to TRL trainer")
                device_info = f"distributed ({accelerator.num_processes} GPUs)"
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
                device_info = str(device)
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            device_info = str(device)

        self.model.eval()
        self.model.config.use_cache = False
        self._reward_fn = get_reward_fn(self.cfg.reward, tokenizer=self.tokenizer)
        logger.info("Model loaded on %s", device_info)
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
        from src.dataset import format_prompt
        import functools

        fmt = functools.partial(format_prompt, tokenizer=self.tokenizer)
        train_dataset = train_dataset.map(fmt, desc="Applying chat template to train")
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(fmt, desc="Applying chat template to eval")

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
            eval_strategy="steps",
            eval_steps=t.eval_steps,
            bf16=self._runtime_bf16,
            fp16=self._runtime_fp16,
            report_to=report_to,
            run_name=self.cfg.condition_id,
            log_completions=True,
        )
        print("=" * 80)
        print(train_dataset[0]["prompt"])
        print("=" * 80)
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
