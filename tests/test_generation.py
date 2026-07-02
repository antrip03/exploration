from __future__ import annotations

from src.config import GenerationConfig
from src.generation import cfg_to_generate_kwargs, detect_device, generate_batch, generate_k_completions, generate_single


def test_cfg_to_generate_kwargs_matches_sampling_mode():
    sampled = cfg_to_generate_kwargs(GenerationConfig())
    greedy = cfg_to_generate_kwargs(GenerationConfig(do_sample=False))
    assert sampled["temperature"] == 0.9
    assert "temperature" not in greedy
    assert sampled["max_new_tokens"] == 512


def test_detect_device_returns_string():
    assert detect_device() in {"cuda", "mps", "cpu"}


def test_generation_helpers_use_model_generate(monkeypatch):
    class DummyTensor:
        shape = (1, 4)

        def to(self, *_args, **_kwargs):
            return self

    class DummyTokenizer:
        pad_token_id = 0
        eos_token = "</s>"
        padding_side = "right"

        def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True):
            return {"input_ids": DummyTensor()}

        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["answer"] * len(sequences)

    class DummyModel:
        def parameters(self):
            return []

        def generate(self, **kwargs):
            import types

            input_ids = kwargs["input_ids"]
            return type("Out", (), {"__getitem__": lambda self, item: self, "shape": (1, 6)})

    tokenizer = DummyTokenizer()
    model = DummyModel()
    cfg = GenerationConfig(max_new_tokens=2, do_sample=False)
    def fake_generate_batch_once(_model, _tokenizer, prompts, cfg, device=None):
        return ["answer"] * (len(prompts) * cfg.num_return_sequences)

    monkeypatch.setattr("src.generation._generate_batch_once", fake_generate_batch_once)
    assert generate_single(model, tokenizer, "prompt", cfg) == "answer"
    assert generate_batch(model, tokenizer, ["a", "b"], cfg, batch_size=1) == ["answer", "answer"]
    assert generate_k_completions(model, tokenizer, ["a"], cfg, k=2) == [["answer", "answer"]]
