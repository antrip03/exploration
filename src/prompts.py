"""
src/prompts.py
==============
Prompt templates for all tasks.

Defines the structured prompt format used throughout training and evaluation.
The format enforces chain-of-thought reasoning via <think> ... </think> tags
and extracts answers via <answer> ... </answer> tags.

Usage:
    from src.prompts import build_countdown_prompt, extract_answer
    prompt = build_countdown_prompt(target=24, numbers=[1, 2, 3, 4])
    answer = extract_answer(model_output)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Template constants
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a mathematical reasoning assistant.
Given a set of numbers and a target, find a way to reach the target
using each number exactly once with basic arithmetic operations (+, -, *, /).

Show your reasoning in <think>...</think> tags, then give your answer in
<answer>...</answer> tags.

Example:
<think>
I have numbers [3, 4, 5] and target 17.
4 * 5 = 20, 20 - 3 = 17.
</think>
<answer>(4 * 5) - 3</answer>"""

COUNTDOWN_TEMPLATE: str = """\
Using the numbers {numbers}, create an arithmetic expression that equals {target}.
Use each number at most once and only the operations +, -, *, /.
Think through the problem carefully, then provide your reasoning and answer using the required tags.
"""

# Regex patterns for parsing model outputs
_THINK_PATTERN: re.Pattern[str] = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
)
_ANSWER_PATTERN: re.Pattern[str] = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParsedOutput:
    """Parsed components of a model output.

    Attributes:
        raw: The raw model output string.
        think_block: Content inside <think>...</think> (None if not found).
        answer_block: Content inside <answer>...</answer> (None if not found).
        has_think: Whether a valid <think> block was found.
        has_answer: Whether a valid <answer> block was found.
        is_well_formed: True if both blocks are present.
    """

    raw: str
    think_block: Optional[str]
    answer_block: Optional[str]

    @property
    def has_think(self) -> bool:
        """Whether a <think> block was extracted."""
        return self.think_block is not None

    @property
    def has_answer(self) -> bool:
        """Whether an <answer> block was extracted."""
        return self.answer_block is not None

    @property
    def is_well_formed(self) -> bool:
        """True if both think and answer blocks are present."""
        return self.has_think and self.has_answer

    @property
    def think_token_count(self) -> int:
        """Approximate token count of the think block (word-based approximation).

        Returns:
            Word count of the think block, or 0 if not present.
        """
        if self.think_block is None:
            return 0
        return len(self.think_block.split())


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────


def build_countdown_prompt(
    target: int,
    numbers: list[int],
    include_system_prompt: bool = False,
) -> str:
    """Build a Countdown task prompt.

    Args:
        target: The target number the model must reach.
        numbers: The list of available numbers to combine.
        include_system_prompt: Whether to prepend the system prompt.
        The default keeps the prompt focused on the task statement so it can be
        wrapped by a chat template later when needed.

    Returns:
        A fully formatted prompt string ready for the model.

    Example:
        >>> prompt = build_countdown_prompt(target=24, numbers=[1, 2, 3, 4])
        >>> print(prompt[:80])
    """
    numbers_str = ", ".join(str(n) for n in numbers)
    body = COUNTDOWN_TEMPLATE.format(numbers=numbers_str, target=target)
    if include_system_prompt:
        return f"{SYSTEM_PROMPT}\n\n{body}"
    return body


def build_chat_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> list[dict[str, str]]:
    """Format a prompt as a HuggingFace chat-style message list.

    Args:
        prompt: The user message content.
        system_prompt: Optional system message. Uses SYSTEM_PROMPT if None.

    Returns:
        A list of message dicts compatible with tokenizer.apply_chat_template().

    Example:
        >>> msgs = build_chat_messages("Solve 2+2")
        >>> tokenizer.apply_chat_template(msgs, tokenize=False)
    """
    sys = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Output parsers
# ─────────────────────────────────────────────────────────────────────────────


def parse_model_output(raw: str) -> ParsedOutput:
    """Parse a raw model output into think and answer components.

    Args:
        raw: The raw string output from the model.

    Returns:
        A ParsedOutput dataclass with extracted blocks.

    Example:
        >>> parsed = parse_model_output("<think>3+1=4</think><answer>4</answer>")
        >>> parsed.answer_block
        '4'
    """
    think_match = _THINK_PATTERN.search(raw)
    answer_match = _ANSWER_PATTERN.search(raw)

    think_block = think_match.group(1).strip() if think_match else None
    answer_block = answer_match.group(1).strip() if answer_match else None

    if think_block is None:
        logger.debug("No <think> block found in model output.")
    if answer_block is None:
        logger.debug("No <answer> block found in model output.")

    return ParsedOutput(
        raw=raw,
        think_block=think_block,
        answer_block=answer_block,
    )


def extract_answer(raw: str) -> Optional[str]:
    """Extract only the answer block from a raw model output.

    A convenience wrapper around parse_model_output.

    Args:
        raw: Raw model output string.

    Returns:
        Stripped answer text, or None if not found.
    """
    return parse_model_output(raw).answer_block


def extract_think(raw: str) -> Optional[str]:
    """Extract only the think block from a raw model output.

    Args:
        raw: Raw model output string.

    Returns:
        Stripped think text, or None if not found.
    """
    return parse_model_output(raw).think_block
