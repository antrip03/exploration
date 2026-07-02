from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.trainer import GRPOExperimentTrainer


def test_instantiation_and_ready_flag(experiment_config):
    trainer = GRPOExperimentTrainer(experiment_config)
    assert trainer.cfg.condition_id == "test_baseline"
    assert trainer.is_ready is False


def test_setup_trainer_requires_loaded_model(experiment_config):
    trainer = GRPOExperimentTrainer(experiment_config)
    with pytest.raises(RuntimeError):
        trainer.setup_trainer(train_dataset=None)


def test_train_requires_setup(experiment_config):
    trainer = GRPOExperimentTrainer(experiment_config)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_load_model_can_be_mocked(experiment_config, monkeypatch):
    trainer = GRPOExperimentTrainer(experiment_config)

    class DummyTokenizer(SimpleNamespace):
        padding_side = "left"
        pad_token_id = None
        eos_token = "</s>"

        def save_pretrained(self, *_args, **_kwargs):
            return None

    class DummyModel(SimpleNamespace):
        config = SimpleNamespace(use_cache=True)

        def parameters(self):
            return []

        def to(self, device):
            return self

        def eval(self):
            return self

        def save_pretrained(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: DummyModel())
    trainer.load_model()
    assert trainer.model is not None
    assert trainer.tokenizer is not None
