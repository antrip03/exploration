"""
tests/test_generation.py
========================
Tests for src/generation.py — text generation utilities.
"""

from __future__ import annotations

import pytest

from src.generation import cfg_to_generate_kwargs, detect_device
from src.config import GenerationConfig


class TestCfgToGenerateKwargs:
    def test_returns_dict(self):
        """cfg_to_generate_kwargs returns a dict."""
        cfg = GenerationConfig()
        kwargs = cfg_to_generate_kwargs(cfg)
        assert isinstance(kwargs, dict)

    def test_contains_required_keys(self):
        """Returned dict contains all expected generate() keys."""
        cfg = GenerationConfig()
        kwargs = cfg_to_generate_kwargs(cfg)
        for key in ("max_new_tokens", "do_sample", "repetition_penalty"):
            assert key in kwargs

    def test_sampling_params_none_when_greedy(self):
        """temperature and top_p are None when do_sample=False."""
        cfg = GenerationConfig(do_sample=False)
        kwargs = cfg_to_generate_kwargs(cfg)
        assert kwargs["temperature"] is None
        assert kwargs["top_p"] is None

    def test_max_new_tokens_correct(self):
        """max_new_tokens matches config."""
        cfg = GenerationConfig(max_new_tokens=256)
        kwargs = cfg_to_generate_kwargs(cfg)
        assert kwargs["max_new_tokens"] == 256


class TestDetectDevice:
    def test_returns_string(self):
        """detect_device returns a non-empty string."""
        device = detect_device()
        assert isinstance(device, str)
        assert len(device) > 0

    def test_returns_valid_device(self):
        """detect_device returns one of the expected device strings."""
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")


class TestGenerateSingle:
    def test_raises_not_implemented(self):
        """generate_single raises NotImplementedError."""
        from src.generation import generate_single
        cfg = GenerationConfig()
        with pytest.raises(NotImplementedError):
            generate_single(None, None, "prompt", cfg)


class TestGenerateBatch:
    def test_raises_not_implemented(self):
        """generate_batch raises NotImplementedError."""
        from src.generation import generate_batch
        cfg = GenerationConfig()
        with pytest.raises(NotImplementedError):
            generate_batch(None, None, ["prompt"], cfg)


class TestGenerateKCompletions:
    def test_raises_not_implemented(self):
        """generate_k_completions raises NotImplementedError."""
        from src.generation import generate_k_completions
        cfg = GenerationConfig()
        with pytest.raises(NotImplementedError):
            generate_k_completions(None, None, ["prompt"], cfg, k=4)
