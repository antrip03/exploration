"""
src/trainer.py
==============
GRPOTrainer wrapper for the reward hacking study.

Wraps HuggingFace TRL GRPOTrainer to provide a clean interface for:
  - Loading the model and tokenizer
  - Applying LoRA via PEFT
  - Wiring up reward functions
  - Running training, evaluation, saving, and loading

No training is implemented in this skeleton. All methods contain TODOs.

Usage:
    from src.trainer import GRPOExperimentTrainer
    from src.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/c1_baseline.yaml")
    trainer = GRPOExperimentTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.config import ExperimentConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.reward_functions import get_reward_fn

logger = logging.getLogger(__name__)


class GRPOExperimentTrainer:
    """Wrapper around HuggingFace TRL GRPOTrainer for GRPO reward hacking experiments.

    Manages the full lifecycle:
      1. Model and tokenizer loading
      2. LoRA adapter application (PEFT)
      3. Dataset wiring
      4. Reward function injection
      5. Training, evaluation, saving, and restoring

    Attributes:
        cfg: Full experiment configuration.
        model: The language model (set after load_model() is called).
        tokenizer: The tokenizer (set after load_model() is called).
        trainer: The underlying TRL GRPOTrainer instance.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        """Initialise the trainer wrapper.

        Args:
            cfg: Validated ExperimentConfig for this experimental condition.
        """
        self.cfg = cfg
        self.model: Any = None        # transformers.PreTrainedModel
        self.tokenizer: Any = None    # transformers.PreTrainedTokenizer
        self.trainer: Any = None      # trl.GRPOTrainer
        self._reward_fn = get_reward_fn(cfg.reward)
        logger.info(
            "GRPOExperimentTrainer initialised — condition: %s", cfg.condition_id
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the base model and tokenizer; apply LoRA if enabled.

        Loads from HuggingFace Hub or local path as specified in cfg.model.
        If cfg.lora.enabled is True, wraps the model with PEFT LoRA config.

        Raises:
            RuntimeError: If model loading fails.

        TODO:
            - Load tokenizer via AutoTokenizer.from_pretrained().
            - Load model via AutoModelForCausalLM.from_pretrained() with
              correct dtype and attn_implementation.
            - Apply get_peft_model() if cfg.lora.enabled.
            - Log model parameter count (trainable vs. total).
            - Handle flash_attention_2 fallback for environments without FA2.
        """
        logger.info("Loading model: %s", self.cfg.model.name)
        # TODO: Implement model loading
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # from peft import get_peft_model, LoraConfig, TaskType
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.cfg.model.name,
        #     revision=self.cfg.model.revision,
        #     trust_remote_code=self.cfg.model.trust_remote_code,
        # )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.cfg.model.name,
        #     torch_dtype=...,
        #     attn_implementation=self.cfg.model.attn_implementation,
        # )
        # if self.cfg.lora.enabled:
        #     lora_config = self._build_lora_config()
        #     self.model = get_peft_model(self.model, lora_config)
        raise NotImplementedError(
            "load_model() is not yet implemented. See TODOs in src/trainer.py."
        )

    def _build_lora_config(self) -> Any:
        """Construct a PEFT LoraConfig from cfg.lora.

        Returns:
            A peft.LoraConfig instance.

        TODO:
            - Import and construct peft.LoraConfig.
            - Map cfg.lora fields to PEFT fields.
        """
        # TODO: Implement LoRA config construction
        # from peft import LoraConfig, TaskType
        # return LoraConfig(
        #     r=self.cfg.lora.r,
        #     lora_alpha=self.cfg.lora.alpha,
        #     lora_dropout=self.cfg.lora.dropout,
        #     target_modules=self.cfg.lora.target_modules,
        #     bias=self.cfg.lora.bias,
        #     task_type=TaskType.CAUSAL_LM,
        # )
        raise NotImplementedError("_build_lora_config() not yet implemented.")

    # ─────────────────────────────────────────────────────────────────────────
    # Trainer setup
    # ─────────────────────────────────────────────────────────────────────────

    def setup_trainer(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> None:
        """Construct the TRL GRPOTrainer with all components.

        Args:
            train_dataset: HuggingFace Dataset for training.
            eval_dataset: HuggingFace Dataset for evaluation (optional).

        Raises:
            RuntimeError: If model has not been loaded via load_model().

        TODO:
            - Build GRPOConfig from cfg.training.
            - Instantiate trl.GRPOTrainer with model, tokenizer, reward_fn,
              train_dataset, eval_dataset, and config.
            - Wire up the reward function signature (TRL expects a specific format).
            - Handle the kl_coef parameter (TRL's built-in KL) vs. manual KL penalty.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before setup_trainer().")
        logger.info("Setting up GRPOTrainer.")
        # TODO: Implement trainer setup
        # from trl import GRPOConfig, GRPOTrainer
        # grpo_cfg = GRPOConfig(
        #     output_dir=self.cfg.training.output_dir,
        #     max_steps=self.cfg.training.max_steps,
        #     per_device_train_batch_size=self.cfg.training.per_device_train_batch_size,
        #     num_generations=self.cfg.training.num_generations,
        #     ...
        # )
        # self.trainer = GRPOTrainer(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     reward_funcs=[self._reward_fn],
        #     args=grpo_cfg,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        # )
        raise NotImplementedError("setup_trainer() not yet implemented.")

    # ─────────────────────────────────────────────────────────────────────────
    # Core lifecycle methods
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the GRPO training loop.

        Raises:
            RuntimeError: If trainer has not been set up via setup_trainer().

        TODO:
            - Call self.trainer.train().
            - Log training metrics at each logging_steps.
            - Handle keyboard interrupt gracefully (save on Ctrl+C).
            - Call save() automatically at the end of training.
        """
        if self.trainer is None:
            raise RuntimeError("Call setup_trainer() before train().")
        logger.info(
            "Starting training — condition: %s, max_steps: %d",
            self.cfg.condition_id,
            self.cfg.training.max_steps,
        )
        # TODO: self.trainer.train()
        raise NotImplementedError("train() not yet implemented.")

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
    ) -> dict[str, float]:
        """Run evaluation on a dataset and return metrics.

        Args:
            eval_dataset: Dataset to evaluate on. Uses the dataset provided to
                setup_trainer() if None.

        Returns:
            Dict mapping metric names to float values.

        TODO:
            - Call self.trainer.evaluate() or run custom evaluation loop.
            - Compute and return MetricsResult via src.metrics.compute_all_metrics().
            - Log results to all configured logging backends.
        """
        logger.info("Running evaluation.")
        # TODO: Implement evaluation
        raise NotImplementedError("evaluate() not yet implemented.")

    def save(self, path: Optional[str] = None) -> None:
        """Save model, tokenizer, and config to disk.

        Args:
            path: Directory to save to. Defaults to cfg.training.output_dir.

        TODO:
            - Save LoRA adapter via model.save_pretrained().
            - Save tokenizer via tokenizer.save_pretrained().
            - Save cfg as config.yaml in the output dir.
            - Log the save path.
        """
        save_path = Path(path or self.cfg.training.output_dir)
        logger.info("Saving model to %s", save_path)
        # TODO: Implement model saving
        raise NotImplementedError("save() not yet implemented.")

    def load(self, path: str) -> None:
        """Load a previously saved model and LoRA adapter from disk.

        Args:
            path: Directory containing saved model files.

        TODO:
            - Load tokenizer from path.
            - Load base model + LoRA weights using PeftModel.from_pretrained().
            - Restore cfg from saved config.yaml if needed.
        """
        logger.info("Loading model from %s", path)
        # TODO: Implement model loading from checkpoint
        # from peft import PeftModel
        # self.model = PeftModel.from_pretrained(base_model, path)
        raise NotImplementedError("load() not yet implemented.")

    # ─────────────────────────────────────────────────────────────────────────
    # Properties / helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """Whether the trainer is fully set up and ready to train.

        Returns:
            True if model, tokenizer, and trainer are all initialised.
        """
        return self.model is not None and self.tokenizer is not None and self.trainer is not None

    def log_model_info(self) -> None:
        """Log model architecture and parameter count information.

        TODO:
            - Count trainable parameters vs. total parameters.
            - Log architecture summary.
        """
        if self.model is None:
            logger.warning("Model not loaded, cannot log model info.")
            return
        # TODO: Log parameter counts
        # trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total = sum(p.numel() for p in self.model.parameters())
        # logger.info("Parameters: %d trainable / %d total (%.1f%%)", ...)
        raise NotImplementedError("log_model_info() not yet implemented.")
