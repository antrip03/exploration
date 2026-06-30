"""
src/generation.py
=================
Text generation utilities for inference and evaluation.

Provides:
  - Single and batched text generation
  - Sampling with temperature / top-p / top-k
  - Greedy decoding
  - GPU-aware batch sizing

Usage:
    from src.generation import generate_batch
    completions = generate_batch(model, tokenizer, prompts, cfg.generation)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.config import GenerationConfig

logger = logging.getLogger(__name__)


def generate_single(
    model: Any,
    tokenizer: Any,
    prompt: str,
    cfg: GenerationConfig,
    device: Optional[str] = None,
) -> str:
    """Generate a single completion for one prompt.

    Args:
        model: Loaded language model (HuggingFace PreTrainedModel).
        tokenizer: Loaded tokenizer.
        prompt: Input prompt string.
        cfg: GenerationConfig parameters.
        device: Device string ('cuda', 'cpu'). Auto-detected if None.

    Returns:
        Generated text string (completion only, not including prompt).

    TODO:
        - Tokenise prompt using tokenizer.apply_chat_template() if chat model.
        - Call model.generate() with cfg parameters.
        - Decode output tokens, strip prompt tokens.
        - Handle padding and attention masks correctly.
    """
    logger.debug("Generating single completion.")
    # TODO: Implement single generation
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, **cfg_to_generate_kwargs(cfg))
    # completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], ...)
    raise NotImplementedError("generate_single() not yet implemented.")


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    cfg: GenerationConfig,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> list[str]:
    """Generate completions for a batch of prompts.

    Handles batching internally with configurable batch size.

    Args:
        model: Loaded language model.
        tokenizer: Loaded tokenizer.
        prompts: List of prompt strings.
        cfg: GenerationConfig parameters.
        batch_size: Number of prompts per forward pass.
        device: Device string. Auto-detected if None.

    Returns:
        List of completion strings, one per prompt (in the same order).

    TODO:
        - Iterate over prompts in batches of batch_size.
        - Pad inputs within each batch.
        - Handle variable-length outputs gracefully.
        - Implement dynamic batch size reduction on OOM.
    """
    logger.info(
        "Generating completions for %d prompts (batch_size=%d).",
        len(prompts),
        batch_size,
    )
    # TODO: Implement batch generation
    raise NotImplementedError("generate_batch() not yet implemented.")


def generate_k_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    cfg: GenerationConfig,
    k: int = 8,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> list[list[str]]:
    """Generate k completions per prompt (for pass@k evaluation).

    Args:
        model: Loaded language model.
        tokenizer: Loaded tokenizer.
        prompts: List of prompt strings.
        cfg: GenerationConfig parameters.
        k: Number of completions per prompt.
        batch_size: GPU batch size per forward pass.
        device: Device string. Auto-detected if None.

    Returns:
        A list of lists: completions[i] is a list of k strings for prompts[i].

    TODO:
        - Run generate_batch() k times or use num_return_sequences=k.
        - Organise outputs per prompt.
        - Ensure reproducibility via seeds per sample.
    """
    logger.info(
        "Generating %d completions for each of %d prompts.", k, len(prompts)
    )
    # TODO: Implement k-completion generation
    raise NotImplementedError("generate_k_completions() not yet implemented.")


def cfg_to_generate_kwargs(cfg: GenerationConfig) -> dict[str, Any]:
    """Convert GenerationConfig to model.generate() keyword arguments.

    Args:
        cfg: GenerationConfig to convert.

    Returns:
        Dict of kwargs compatible with HuggingFace model.generate().
    """
    return {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature if cfg.do_sample else None,
        "top_p": cfg.top_p if cfg.do_sample else None,
        "top_k": cfg.top_k if cfg.do_sample else None,
        "repetition_penalty": cfg.repetition_penalty,
        "num_return_sequences": cfg.num_return_sequences,
    }


def detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        'cuda' if a CUDA GPU is available, 'mps' for Apple Silicon, else 'cpu'.

    TODO:
        - Handle multi-GPU selection (e.g., pick device with most free memory).
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Detected device: %s", device)
        return device
    except ImportError:
        logger.warning("torch not installed, defaulting to 'cpu'.")
        return "cpu"
