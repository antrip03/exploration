from __future__ import annotations

from src.dataset import collate_fn, format_prompt, preprocess_countdown_example


def test_preprocess_countdown_example_returns_expected_keys():
    row = {"nums": [1, 2, 3, 4], "target": 24}
    processed = preprocess_countdown_example(row)
    assert processed["answer"] == "24"
    assert processed["target"] == 24
    assert processed["nums"] == [1, 2, 3, 4]
    assert "24" in processed["prompt"]


def test_format_prompt_uses_chat_template():
    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "|".join(message["role"] + ":" + message["content"] for message in messages)

    result = format_prompt({"prompt": "Solve 2+2", "answer": "4"}, DummyTokenizer())
    assert result["prompt"].startswith("system:")
    assert "user:Solve 2+2" in result["prompt"]


def test_collate_fn_batches_rows():
    batch = [{"prompt": "p1", "answer": "a1"}, {"prompt": "p2", "answer": "a2"}]
    out = collate_fn(batch)
    assert out["prompt"] == ["p1", "p2"]
    assert out["answer"] == ["a1", "a2"]
