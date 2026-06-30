"""
tests/test_prompts.py
=====================
Tests for src/prompts.py — prompt building and output parsing.
"""

from __future__ import annotations

import pytest

from src.prompts import (
    ParsedOutput,
    build_chat_messages,
    build_countdown_prompt,
    build_gsm8k_prompt,
    extract_answer,
    extract_think,
    parse_model_output,
)


class TestBuildCountdownPrompt:
    def test_contains_target(self):
        """Prompt contains the target number."""
        prompt = build_countdown_prompt(target=24, numbers=[1, 2, 3, 4])
        assert "24" in prompt

    def test_contains_all_numbers(self):
        """Prompt contains all available numbers."""
        numbers = [1, 2, 3, 4, 6, 8]
        prompt = build_countdown_prompt(target=24, numbers=numbers)
        for n in numbers:
            assert str(n) in prompt

    def test_contains_think_tags(self):
        """Prompt template contains <think> and </think> markers."""
        prompt = build_countdown_prompt(target=10, numbers=[2, 5])
        assert "<think>" in prompt
        assert "</think>" in prompt

    def test_contains_answer_tags(self):
        """Prompt template contains <answer> and </answer> markers."""
        prompt = build_countdown_prompt(target=10, numbers=[2, 5])
        assert "<answer>" in prompt
        assert "</answer>" in prompt

    def test_with_system_prompt(self):
        """System prompt is included when include_system_prompt=True."""
        prompt = build_countdown_prompt(target=10, numbers=[5, 2], include_system_prompt=True)
        assert len(prompt) > 0
        # System prompt mentions reasoning
        assert "reasoning" in prompt.lower() or "think" in prompt.lower()

    def test_without_system_prompt(self):
        """System prompt is excluded when include_system_prompt=False."""
        prompt_with = build_countdown_prompt(target=10, numbers=[5, 2], include_system_prompt=True)
        prompt_without = build_countdown_prompt(target=10, numbers=[5, 2], include_system_prompt=False)
        assert len(prompt_with) > len(prompt_without)

    def test_empty_numbers_list(self):
        """Prompt handles an empty numbers list gracefully."""
        prompt = build_countdown_prompt(target=0, numbers=[])
        assert isinstance(prompt, str)


class TestBuildGSM8KPrompt:
    def test_contains_problem(self):
        """Prompt contains the problem text."""
        problem = "Janet has 3 apples and buys 5 more. How many does she have?"
        prompt = build_gsm8k_prompt(problem=problem)
        assert "Janet" in prompt

    def test_contains_think_tags(self):
        """GSM8K prompt contains think/answer structure."""
        prompt = build_gsm8k_prompt(problem="2+2=?")
        assert "<think>" in prompt
        assert "<answer>" in prompt


class TestBuildChatMessages:
    def test_returns_list_of_dicts(self):
        """chat messages is a list of role/content dicts."""
        msgs = build_chat_messages("Hello")
        assert isinstance(msgs, list)
        assert all("role" in m and "content" in m for m in msgs)

    def test_has_system_and_user_roles(self):
        """Messages contain both system and user roles."""
        msgs = build_chat_messages("Solve 2+2")
        roles = {m["role"] for m in msgs}
        assert "system" in roles
        assert "user" in roles

    def test_user_content_matches(self):
        """User message content matches provided prompt."""
        msgs = build_chat_messages("My prompt")
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert user_msg["content"] == "My prompt"


class TestParseModelOutput:
    def test_well_formed_output(self, well_formed_completion):
        """Well-formed output is parsed correctly."""
        parsed = parse_model_output(well_formed_completion)
        assert parsed.has_think
        assert parsed.has_answer
        assert parsed.is_well_formed

    def test_missing_answer(self, missing_answer_completion):
        """Output with only think block is parsed correctly."""
        parsed = parse_model_output(missing_answer_completion)
        assert parsed.has_think
        assert not parsed.has_answer
        assert not parsed.is_well_formed

    def test_malformed_completion(self, malformed_completion):
        """Malformed output returns None for both blocks."""
        parsed = parse_model_output(malformed_completion)
        assert not parsed.has_think
        assert not parsed.has_answer
        assert not parsed.is_well_formed

    def test_think_token_count_zero_when_missing(self, malformed_completion):
        """think_token_count returns 0 when no think block."""
        parsed = parse_model_output(malformed_completion)
        assert parsed.think_token_count == 0

    def test_think_token_count_positive(self, well_formed_completion):
        """think_token_count is positive for a non-empty think block."""
        parsed = parse_model_output(well_formed_completion)
        assert parsed.think_token_count > 0

    def test_raw_preserved(self, well_formed_completion):
        """Raw string is preserved unchanged."""
        parsed = parse_model_output(well_formed_completion)
        assert parsed.raw == well_formed_completion


class TestExtractHelpers:
    def test_extract_answer_found(self, well_formed_completion):
        """extract_answer returns the answer block when present."""
        answer = extract_answer(well_formed_completion)
        assert answer is not None
        assert len(answer) > 0

    def test_extract_answer_not_found(self, malformed_completion):
        """extract_answer returns None when not found."""
        answer = extract_answer(malformed_completion)
        assert answer is None

    def test_extract_think_found(self, well_formed_completion):
        """extract_think returns the think block when present."""
        think = extract_think(well_formed_completion)
        assert think is not None
        assert len(think) > 0

    def test_extract_think_not_found(self, malformed_completion):
        """extract_think returns None when not found."""
        think = extract_think(malformed_completion)
        assert think is None
