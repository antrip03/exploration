"""Batched Hugging Face generation helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.config import GenerationConfig

logger = logging.getLogger(__name__)


def _model_device(model: Any, requested: Optional[str]) -> Any:
    if requested is not None:
        return requested
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return detect_device()


def cfg_to_generate_kwargs(cfg: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "repetition_penalty": cfg.repetition_penalty,
        "num_return_sequences": cfg.num_return_sequences,
    }
    if cfg.do_sample:
        kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
    return kwargs


def _generate_batch_once(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    cfg: GenerationConfig,
    device: Optional[str] = None,
) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(_model_device(model, device)) for key, value in encoded.items()}
    kwargs = cfg_to_generate_kwargs(cfg)
    kwargs.update(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **kwargs)

    prompt_width = encoded["input_ids"].shape[1]
    generated = output_ids[:, prompt_width:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def generate_single(
    model: Any,
    tokenizer: Any,
    prompt: str,
    cfg: GenerationConfig,
    device: Optional[str] = None,
) -> str:
    """Generate one completion without returning prompt tokens."""
    single_cfg = cfg.model_copy(update={"num_return_sequences": 1})
    return _generate_batch_once(model, tokenizer, [prompt], single_cfg, device)[0]


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    cfg: GenerationConfig,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> list[str]:
    """Generate ``num_return_sequences`` outputs for each input prompt."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        try:
            outputs.extend(_generate_batch_once(model, tokenizer, batch, cfg, device))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size == 1:
                raise
            logger.warning("Generation OOM at batch_size=%d; retrying at half size", batch_size)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            outputs.extend(
                generate_batch(
                    model,
                    tokenizer,
                    batch,
                    cfg,
                    batch_size=max(1, batch_size // 2),
                    device=device,
                )
            )
    return outputs


def generate_k_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    cfg: GenerationConfig,
    k: int = 8,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> list[list[str]]:
    """Sample exactly ``k`` completions per prompt."""
    if k < 1:
        raise ValueError("k must be positive")
    sample_cfg = cfg.model_copy(
        update={"do_sample": True, "num_return_sequences": k}
    )
    flat = generate_batch(model, tokenizer, prompts, sample_cfg, batch_size, device)
    expected = len(prompts) * k
    if len(flat) != expected:
        raise RuntimeError(f"Expected {expected} generated sequences, got {len(flat)}")
    return [flat[index * k : (index + 1) * k] for index in range(len(prompts))]


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
