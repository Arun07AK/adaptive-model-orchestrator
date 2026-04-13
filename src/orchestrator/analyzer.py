from __future__ import annotations

import re

from src.types import Complexity, Domain, TaskAnalysis

_MATH_SUBJECTS = {
    "abstract_algebra", "college_mathematics", "elementary_mathematics",
    "high_school_mathematics", "high_school_statistics",
    "econometrics",
}

_CODE_SUBJECTS = {
    "college_computer_science", "high_school_computer_science",
}

_REASONING_SUBJECTS = {
    "logical_fallacies", "formal_logic", "moral_disputes", "moral_scenarios",
}

_MATH_PATTERNS = re.compile(
    r"\b(solve|equation|integral|derivative|matrix|eigenvalue|probability|"
    r"calculate|compute|sum|product|factorial|theorem|proof|limit|"
    r"polynomial|logarithm|trigonometric|algebra|geometry|calculus|"
    r"denominator|numerator|fraction|sqrt|sin|cos|tan|pi|"
    r"\d+\s*[\+\-\*/\^]\s*\d+)\b",
    re.IGNORECASE,
)

_CODE_PATTERNS = re.compile(
    r"\b(function|def |class |import |return |variable|algorithm|"
    r"implement|code|program|python|java|javascript|array|list|"
    r"linked.list|binary.tree|hash.map|stack|queue|loop|recursion|"
    r"compile|runtime|syntax|API|database|SQL|git|debug)\b",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"\b(if all|conclude|therefore|implies|logically|argument|"
    r"premise|valid|fallacy|infer|deduce|syllogism|contradiction|"
    r"necessary|sufficient)\b",
    re.IGNORECASE,
)


class TaskAnalyzer:
    def classify(
        self,
        text: str,
        subject_hint: str | None = None,
    ) -> TaskAnalysis:
        domain, confidence = self._detect_domain(text, subject_hint)
        complexity = self._detect_complexity(text)

        return TaskAnalysis(
            text=text,
            domain=domain,
            complexity=complexity,
            confidence=confidence,
        )

    def _detect_domain(
        self, text: str, subject_hint: str | None
    ) -> tuple[Domain, float]:
        if subject_hint:
            subject = subject_hint.lower().replace(" ", "_")
            if subject in _MATH_SUBJECTS:
                return Domain.MATH, 0.95
            if subject in _CODE_SUBJECTS:
                return Domain.CODE, 0.95
            if subject in _REASONING_SUBJECTS:
                return Domain.REASONING, 0.90
            return Domain.GENERAL, 0.85

        math_hits = len(_MATH_PATTERNS.findall(text))
        code_hits = len(_CODE_PATTERNS.findall(text))
        reasoning_hits = len(_REASONING_PATTERNS.findall(text))

        scores = {
            Domain.MATH: math_hits * 1.5,
            Domain.CODE: code_hits * 1.5,
            Domain.REASONING: reasoning_hits * 1.2,
            Domain.GENERAL: 0.5,
        }

        best_domain = max(scores, key=lambda d: scores[d])
        total = sum(scores.values()) or 1.0
        confidence = scores[best_domain] / total

        return best_domain, min(confidence, 1.0)

    def _detect_complexity(self, text: str) -> Complexity:
        if len(text) > 500:
            return Complexity.COMPLEX
        multi_step_indicators = ["step 1", "first,", "then,", "finally,", "part a", "part b"]
        if any(indicator in text.lower() for indicator in multi_step_indicators):
            return Complexity.COMPLEX
        return Complexity.SIMPLE
