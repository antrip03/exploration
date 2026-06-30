"""
tests/test_dataset.py
=====================
Tests for src/dataset.py — dataset loaders and preprocessing.
"""

from __future__ import annotations

import pytest

from src.dataset import (
    collate_fn,
    generate_countdown_examples,
    preprocess_countdown_example,
    preprocess_gsm8k_example,
)


class TestGenerateCountdownExamples:
    def test_returns_correct_count(self):
        """Generates exactly n_examples examples.

        TODO: Enable once generate_countdown_examples() is fully implemented.
        """
        pytest.skip("TODO: Implement generate_countdown_examples()")

    def test_reproducible_with_seed(self):
        """Same seed produces same examples.

        TODO: Enable once implemented.
        """
        pytest.skip("TODO: Test reproducibility with same seed")

    def test_num_numbers_correct(self):
        """Each example has the correct number of source numbers.

        TODO: Enable once implemented.
        """
        pytest.skip("TODO: Test num_numbers constraint")


class TestPreprocessCountdownExample:
    def test_returns_expected_keys(self):
        """Preprocessed example has prompt, answer, metadata keys.

        TODO: Implement once field mapping is known.
        """
        pytest.skip("TODO: Implement once HF dataset schema is confirmed")

    def test_prompt_contains_target(self):
        """Preprocessed prompt contains the target number.

        TODO: Enable once implemented.
        """
        pytest.skip("TODO: Test prompt content")


class TestPreprocessGSM8KExample:
    def test_returns_expected_keys(self):
        """GSM8K preprocessed example has correct keys.

        TODO: Enable once GSM8K loading is implemented.
        """
        pytest.skip("TODO: Implement GSM8K preprocessing tests")

    def test_numeric_answer_extracted(self):
        """Numeric answer is correctly extracted after ####.

        TODO: Enable once extraction is implemented.
        """
        pytest.skip("TODO: Test GSM8K answer extraction")


class TestCollateFn:
    def test_batches_correctly(self):
        """collate_fn correctly batches a list of dicts."""
        batch = [
            {"prompt": "p1", "answer": "a1"},
            {"prompt": "p2", "answer": "a2"},
        ]
        result = collate_fn(batch)
        assert result["prompt"] == ["p1", "p2"]
        assert result["answer"] == ["a1", "a2"]

    def test_empty_batch(self):
        """collate_fn handles empty batch gracefully.

        TODO: Decide expected behaviour for empty batch.
        """
        pytest.skip("TODO: Define empty batch behaviour")
