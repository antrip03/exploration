from __future__ import annotations

from src.prompts import build_chat_messages, build_countdown_prompt, extract_answer, extract_think, parse_model_output


def test_build_countdown_prompt_mentions_problem():
    prompt = build_countdown_prompt(target=24, numbers=[1, 2, 3, 4])
    assert "24" in prompt
    assert "1, 2, 3, 4" in prompt
    assert "<think>" not in prompt


def test_build_chat_messages_contains_system_and_user():
    messages = build_chat_messages("Solve 2+2")
    roles = {message["role"] for message in messages}
    assert roles == {"system", "user"}


def test_parse_model_output_extracts_blocks(well_formed_completion):
    parsed = parse_model_output(well_formed_completion)
    assert parsed.has_think
    assert parsed.has_answer
    assert parsed.is_well_formed
    assert parsed.think_token_count > 0
    assert extract_answer(well_formed_completion) is not None
    assert extract_think(well_formed_completion) is not None
