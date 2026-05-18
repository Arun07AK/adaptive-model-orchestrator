from __future__ import annotations

import re

_NUMBER_RE = r"-?\d+(?:\.\d+)?"


def strip_reasoning(text: str) -> str:
    """Remove model reasoning blocks before extracting final answers."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def extract_choice_answer(text: str, *, include_fallback: bool = True) -> str | None:
    """Extract an A-D multiple-choice answer, preferring explicit final markers."""
    clean = strip_reasoning(text)
    stripped = clean.strip().rstrip(".").rstrip(")").strip()
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()

    upper = clean.upper()
    for pattern in [
        r"(?:CORRECT ANSWER|FINAL ANSWER|THE ANSWER)\s*(?:IS|:)\s*\*?\*?([A-D])",
        r"ANSWER\s*(?:IS|:)\s*\*?\*?([A-D])",
        r"\*\*([A-D])\*\*",
        r"(?:OPTION|CHOICE)\s*([A-D])\b",
        r"\b(?:CHOOSE|SELECT|PICK)\s*(?:OPTION|CHOICE)?\s*([A-D])\b",
        r"^\s*\(?([A-D])[\.\)]",
        r"^\s*([A-D])\s*$",
    ]:
        match = re.search(pattern, upper, re.MULTILINE)
        if match:
            return match.group(1)

    if not include_fallback:
        return None

    matches = re.findall(r"\b([A-D])\b", upper)
    return matches[-1] if matches else None


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from a model response."""
    clean = strip_reasoning(text)
    clean_no_commas = clean.replace(",", "")

    hash_matches = re.findall(rf"####\s*({_NUMBER_RE})", clean_no_commas)
    if hash_matches:
        return hash_matches[-1]

    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", clean_no_commas)
    if boxed_matches:
        boxed_numbers = re.findall(_NUMBER_RE, boxed_matches[-1])
        if boxed_numbers:
            return boxed_numbers[-1]

    answer_matches = re.findall(
        rf"(?:FINAL ANSWER|THE ANSWER|ANSWER)\s*(?:IS|:)\s*({_NUMBER_RE})",
        clean_no_commas,
        re.IGNORECASE,
    )
    if answer_matches:
        return answer_matches[-1]

    if "</think>" in text:
        after_think = text.split("</think>")[-1].replace(",", "")
        after_think_numbers = re.findall(_NUMBER_RE, after_think)
        if after_think_numbers:
            return after_think_numbers[-1]

    numbers = re.findall(_NUMBER_RE, clean_no_commas)
    return numbers[-1] if numbers else None


def normalize_answer(text: str) -> str:
    """Normalize a model answer for consistency checks."""
    explicit_choice = extract_choice_answer(text, include_fallback=False)
    if explicit_choice is not None:
        return explicit_choice

    number = extract_numeric_answer(text)
    if number is not None:
        return number

    choice = extract_choice_answer(text)
    if choice is not None:
        return choice

    return strip_reasoning(text)[:50].lower()
