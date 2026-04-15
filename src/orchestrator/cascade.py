from __future__ import annotations

import re
from src.orchestrator.executor import Executor
from src.types import ExecutionResult, ModelConfig, OrchestratorResult

_REVIEW_PROMPT = """You are reviewing answers from other models for this question:

Question: {question}

Previous attempts:
{attempts}

The previous models gave different answers. Provide the CORRECT answer.
Answer directly and concisely - do not explain unless required by the question format."""


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison: strip <think> tags, extract letter/number."""
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    # Try to find a letter answer first
    letter = re.search(r'\b([A-D])\b', text.upper())
    if letter:
        return letter.group(1)
    # Try a number
    num = re.search(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    if num:
        return num.group(0)
    return text.strip()[:50].lower()


class SelfConsistencyScorer:
    """Generates 2 answers from a model, checks if they agree."""

    def __init__(self, executor: Executor) -> None:
        self._executor = executor

    async def score(
        self, model: ModelConfig, prompt: str, max_tokens: int = 256,
    ) -> tuple[list[ExecutionResult], bool]:
        """Returns (attempts, is_consistent). Is_consistent if both normalize to same answer."""
        r1 = await self._executor.execute(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
        )
        r2 = await self._executor.execute(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.5,
        )
        return [r1, r2], _normalize_answer(r1.text) == _normalize_answer(r2.text)


class CrossModelConsistencyScorer:
    """Runs TWO DIFFERENT models in parallel, checks if they agree.

    Motivation (from Sushant Gagneja's critique):
    Self-consistency with the same model tests stochastic variance, not
    competence. A model with a systematic knowledge gap will generate the
    same wrong answer at both temperatures.

    Cross-model consistency uses architecturally-orthogonal models (e.g., Llama
    family + Qwen family). They have different training data, different
    failure modes, so agreement between them is a much stronger signal of
    correctness than self-agreement.
    """

    def __init__(self, executor: Executor) -> None:
        self._executor = executor

    async def score(
        self, model_a: ModelConfig, model_b: ModelConfig,
        prompt: str, max_tokens: int = 256,
    ) -> tuple[list[ExecutionResult], bool]:
        """Returns (attempts, is_consistent). Both models at temp=0 (deterministic)."""
        r_a = await self._executor.execute(
            model=model_a, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
        )
        r_b = await self._executor.execute(
            model=model_b, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
        )
        return [r_a, r_b], _normalize_answer(r_a.text) == _normalize_answer(r_b.text)


class CrossModelPipeline:
    """V3: Cross-model consistency. Two architecturally-orthogonal models
    (Llama + Qwen family) answer in parallel. Escalate only if they disagree.

    Much stronger confidence signal than same-model self-consistency because
    the two models have different failure modes — a systematic knowledge gap
    in one is unlikely to be shared by the other.
    """

    def __init__(
        self,
        executor: Executor,
        model_a: ModelConfig,  # e.g., Llama-3.3-70B (Llama family)
        model_b: ModelConfig,  # e.g., Qwen3-32B (Qwen family)
        senior_reviewer: ModelConfig,
        analyzer,
    ) -> None:
        self._executor = executor
        self._model_a = model_a
        self._model_b = model_b
        self._senior = senior_reviewer
        self._analyzer = analyzer
        self._scorer = CrossModelConsistencyScorer(executor)
        self.review_count = 0
        self.total_count = 0

    async def run(
        self, prompt: str, subject_hint: str | None = None, max_tokens: int = 256,
    ) -> OrchestratorResult:
        self.total_count += 1
        # Subject hint used for logging consistency with other pipelines
        _ = self._analyzer.classify(prompt, subject_hint=subject_hint)

        attempts, consistent = await self._scorer.score(
            self._model_a, self._model_b, prompt, max_tokens,
        )

        if consistent:
            # Both models agree - high confidence
            total_latency = attempts[0].latency_ms + attempts[1].latency_ms
            return OrchestratorResult(
                text=attempts[0].text,
                model_used=f"{self._model_a.name} + {self._model_b.name}",
                escalated=False,
                total_latency_ms=total_latency,
                confidence=1.0,
            )

        # Models disagree - escalate to senior
        self.review_count += 1
        attempts_text = "\n".join(
            f"Model {attempts[i].model_used}: {_normalize_answer(a.text)}"
            for i, a in enumerate(attempts)
        )
        review_prompt = _REVIEW_PROMPT.format(question=prompt, attempts=attempts_text)
        review = await self._executor.execute(
            model=self._senior, prompt=review_prompt, max_tokens=max_tokens,
        )
        total_latency = sum(a.latency_ms for a in attempts) + review.latency_ms
        return OrchestratorResult(
            text=review.text,
            model_used=f"{self._model_a.name} + {self._model_b.name} -> {self._senior.name}",
            escalated=True,
            escalation_model=self._senior.name,
            total_latency_ms=total_latency,
            confidence=0.5,
        )


class SelectiveReviewPipeline:
    """Specialist answers. If uncertain (self-consistency fails), Senior reviewer corrects."""

    def __init__(
        self,
        executor: Executor,
        specialist_selector,  # callable(domain) -> ModelConfig
        senior_reviewer: ModelConfig,
        analyzer,  # TaskAnalyzer
    ) -> None:
        self._executor = executor
        self._select_specialist = specialist_selector
        self._senior = senior_reviewer
        self._analyzer = analyzer
        self._scorer = SelfConsistencyScorer(executor)
        self.review_count = 0
        self.total_count = 0

    async def run(
        self, prompt: str, subject_hint: str | None = None, max_tokens: int = 256,
    ) -> OrchestratorResult:
        self.total_count += 1
        analysis = self._analyzer.classify(prompt, subject_hint=subject_hint)
        specialist = self._select_specialist(analysis.domain)

        attempts, consistent = await self._scorer.score(specialist, prompt, max_tokens)

        if consistent:
            # High confidence - return specialist answer
            total_latency = attempts[0].latency_ms + attempts[1].latency_ms
            return OrchestratorResult(
                text=attempts[0].text,
                model_used=specialist.name,
                escalated=False,
                total_latency_ms=total_latency,
                confidence=1.0,
            )

        # Disagree - senior reviews
        self.review_count += 1
        attempts_text = "\n".join(
            f"Attempt {i+1}: {_normalize_answer(a.text)}" for i, a in enumerate(attempts)
        )
        review_prompt = _REVIEW_PROMPT.format(question=prompt, attempts=attempts_text)
        review = await self._executor.execute(
            model=self._senior, prompt=review_prompt, max_tokens=max_tokens,
        )
        total_latency = sum(a.latency_ms for a in attempts) + review.latency_ms
        return OrchestratorResult(
            text=review.text,
            model_used=f"{specialist.name} + {self._senior.name}",
            escalated=True,
            escalation_model=self._senior.name,
            total_latency_ms=total_latency,
            confidence=0.5,
        )


class CascadePipeline:
    """3-Tier: Laborer -> Specialist -> Senior. Escalates on inconsistency."""

    def __init__(
        self,
        executor: Executor,
        laborer: ModelConfig,
        specialist_selector,
        senior_reviewer: ModelConfig,
        analyzer,
    ) -> None:
        self._executor = executor
        self._laborer = laborer
        self._select_specialist = specialist_selector
        self._senior = senior_reviewer
        self._analyzer = analyzer
        self._scorer = SelfConsistencyScorer(executor)
        self.laborer_count = 0
        self.specialist_count = 0
        self.senior_count = 0
        self.total_count = 0

    async def run(
        self, prompt: str, subject_hint: str | None = None, max_tokens: int = 256,
    ) -> OrchestratorResult:
        self.total_count += 1
        analysis = self._analyzer.classify(prompt, subject_hint=subject_hint)

        # Tier 1: Laborer
        self.laborer_count += 1
        lab_attempts, lab_consistent = await self._scorer.score(
            self._laborer, prompt, max_tokens,
        )
        if lab_consistent:
            return OrchestratorResult(
                text=lab_attempts[0].text,
                model_used=self._laborer.name,
                escalated=False,
                total_latency_ms=lab_attempts[0].latency_ms + lab_attempts[1].latency_ms,
                confidence=1.0,
            )

        # Tier 2: Specialist
        self.specialist_count += 1
        specialist = self._select_specialist(analysis.domain)
        if specialist.name == self._laborer.name:
            # Skip if specialist == laborer
            spec_attempts = lab_attempts
            spec_consistent = lab_consistent
        else:
            spec_attempts, spec_consistent = await self._scorer.score(
                specialist, prompt, max_tokens,
            )
        if spec_consistent:
            total_latency = sum(a.latency_ms for a in lab_attempts + spec_attempts)
            return OrchestratorResult(
                text=spec_attempts[0].text,
                model_used=f"{self._laborer.name} -> {specialist.name}",
                escalated=True,
                escalation_model=specialist.name,
                total_latency_ms=total_latency,
                confidence=0.8,
            )

        # Tier 3: Senior reviewer
        self.senior_count += 1
        all_attempts = lab_attempts + spec_attempts
        attempts_text = "\n".join(
            f"Attempt {i+1} ({all_attempts[i].model_used}): {_normalize_answer(a.text)}"
            for i, a in enumerate(all_attempts)
        )
        review_prompt = _REVIEW_PROMPT.format(question=prompt, attempts=attempts_text)
        review = await self._executor.execute(
            model=self._senior, prompt=review_prompt, max_tokens=max_tokens,
        )
        total_latency = sum(a.latency_ms for a in all_attempts) + review.latency_ms
        return OrchestratorResult(
            text=review.text,
            model_used=f"{self._laborer.name} -> {specialist.name} -> {self._senior.name}",
            escalated=True,
            escalation_model=self._senior.name,
            total_latency_ms=total_latency,
            confidence=0.5,
        )
