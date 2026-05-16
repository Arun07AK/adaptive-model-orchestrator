from __future__ import annotations

import re


def strip_reasoning_blocks(text: str) -> str:
    """Remove hidden reasoning blocks emitted by reasoning models."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _extract_choice_answer(text: str, *, include_fallback: bool) -> str | None:
    """Extract a final A-D multiple-choice answer from model output."""
    text = strip_reasoning_blocks(text)

    stripped = text.strip().rstrip(".").rstrip(")").strip()
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()

    upper_text = text.upper()

    for pattern in [
        r"(?:CORRECT ANSWER|FINAL ANSWER|THE ANSWER)\s*(?:IS|:)\s*\*?\*?([A-D])",
        r"ANSWER\s*(?:IS|:)\s*\*?\*?([A-D])",
        r"\*\*([A-D])\*\*",
        r"(?:OPTION|CHOICE)\s*([A-D])\b",
        r"^\s*\(?([A-D])[\.\)]",
        r"^\s*([A-D])\s*$",
    ]:
        match = re.search(pattern, upper_text, re.MULTILINE)
        if match:
            return match.group(1)

    if include_fallback:
        last_match = re.findall(r"\b([A-D])\b", upper_text)
        if last_match:
            return last_match[-1]
    return None


def extract_choice_answer(text: str) -> str | None:
    """Extract a final A-D answer, including fallback heuristics for MCQ scoring."""
    return _extract_choice_answer(text, include_fallback=True)


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from math-style model output."""
    clean = strip_reasoning_blocks(text)

    hash_match = re.findall(
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        clean.replace(",", ""),
    )
    if hash_match:
        return hash_match[-1]

    answer_match = re.search(
        r"(?:the answer is|answer is|answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        clean,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).replace(",", "")

    boxed = re.findall(r"\\boxed\{([^}]+)\}", clean)
    if boxed:
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed[-1].replace(",", ""))
        if nums:
            return nums[-1]

    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", after_think.replace(",", ""))
        if numbers:
            return numbers[-1]

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", clean.replace(",", ""))
    return numbers[-1] if numbers else None


def normalize_answer_for_consistency(text: str) -> str:
    """Normalize an answer for self/cross-model consistency comparisons."""
    choice = _extract_choice_answer(text, include_fallback=False)
    if choice is not None:
        return choice

    number = extract_numeric_answer(text)
    if number is not None:
        return number

    choice = _extract_choice_answer(text, include_fallback=True)
    if choice is not None:
        return choice

    return strip_reasoning_blocks(text).strip()[:50].lower()
