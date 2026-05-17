from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_NUMBER_RE = r"-?\d+(?:,\d{3})*(?:\.\d+)?"


def strip_think_blocks(text: str) -> str:
    """Remove model reasoning blocks while preserving the final response."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def extract_choice_answer(text: str) -> str | None:
    """Extract the intended final A-D answer from a multiple-choice response."""
    clean = strip_think_blocks(text)
    stripped = clean.strip().rstrip(".").rstrip(")").strip()
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()

    upper = clean.upper()
    for pattern in [
        r"(?:CORRECT ANSWER|FINAL ANSWER|THE ANSWER)\s*(?:IS|:)\s*\*?\*?\(?([A-D])\)?",
        r"ANSWER\s*(?:IS|:)\s*\*?\*?\(?([A-D])\)?",
        r"(?:CHOOSE|SELECT|PICK)\s+(?:OPTION|CHOICE)?\s*\(?([A-D])\)?\b",
        r"\*\*([A-D])\*\*",
        r"^\s*\(?([A-D])[\.\)]",
        r"^\s*([A-D])\s*$",
    ]:
        match = re.search(pattern, upper, re.MULTILINE)
        if match:
            return match.group(1)

    last_match = re.findall(r"\b([A-D])\b", upper)
    if last_match:
        return last_match[-1]
    return None


def _numbers(text: str) -> list[str]:
    return [number.replace(",", "") for number in re.findall(_NUMBER_RE, text)]


def _extract_hash_answer(text: str) -> str | None:
    matches = re.findall(rf"####\s*({_NUMBER_RE})", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def _extract_boxed_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not boxed:
        return None
    numbers = _numbers(boxed[-1])
    return numbers[-1] if numbers else None


def _extract_answer_statement(text: str) -> str | None:
    match = re.search(
        rf"(?:the answer is|answer is|answer:)\s*({_NUMBER_RE})",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_numeric_answer(text: str) -> str | None:
    """Extract the intended final numeric answer from a math response."""
    hash_answer = _extract_hash_answer(text)
    if hash_answer is not None:
        return hash_answer

    clean = strip_think_blocks(text)
    if "</think>" in text.lower():
        after_think = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
        for extractor in (_extract_hash_answer, _extract_answer_statement, _extract_boxed_answer):
            answer = extractor(after_think)
            if answer is not None:
                return answer
        numbers = _numbers(after_think)
        if numbers:
            return numbers[-1]

    for source in (clean, text):
        for extractor in (_extract_boxed_answer, _extract_answer_statement):
            answer = extractor(source)
            if answer is not None:
                return answer

    numbers = _numbers(clean)
    return numbers[-1] if numbers else None


def normalize_answer(text: str) -> str:
    """Normalize a model answer for consistency checks and review prompts."""
    choice = extract_choice_answer(text)
    if choice is not None:
        return choice

    number = extract_numeric_answer(text)
    if number is not None:
        return number

    return strip_think_blocks(text)[:50].lower()
