import pytest

from src.orchestrator.cascade import (
    CascadePipeline,
    SelfConsistencyScorer,
    SelectiveReviewPipeline,
    _normalize_answer,
)
from src.orchestrator.executor import Executor
from src.orchestrator.analyzer import TaskAnalyzer
from src.types import CostTier, Domain, ExecutionResult, ModelConfig
from tests.conftest import MockBackend


def _make_model(name, domain=Domain.GENERAL, size=7.0):
    return ModelConfig(
        name=name, provider="mock", domain=domain, size_b=size,
        cost_tier=CostTier.FREE_API, model_id=name,
    )


def test_normalize_answer_letter():
    assert _normalize_answer("The answer is B.") == "B"
    assert _normalize_answer("<think>reasoning</think>A") == "A"


def test_normalize_answer_number():
    assert _normalize_answer("The answer is 42") == "42"
    assert _normalize_answer("#### 72") == "72"


@pytest.mark.asyncio
async def test_self_consistency_agreement():
    backend = MockBackend(default_response="B", default_confidence=0.9)
    executor = Executor(backends={"mock": backend})
    scorer = SelfConsistencyScorer(executor)
    attempts, consistent = await scorer.score(_make_model("m1"), "q?", max_tokens=10)
    assert consistent is True
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_self_consistency_disagreement():
    backend = MockBackend()
    call = [0]
    async def varied(model, prompt, max_tokens=256, temperature=0.0):
        from src.types import ExecutionResult
        call[0] += 1
        text = "A" if call[0] == 1 else "B"
        return ExecutionResult(text=text, confidence=0.5, model_used=model.name, latency_ms=10, token_count=1)
    backend.generate = varied
    executor = Executor(backends={"mock": backend})
    scorer = SelfConsistencyScorer(executor)
    attempts, consistent = await scorer.score(_make_model("m1"), "q?", max_tokens=10)
    assert consistent is False


@pytest.mark.asyncio
async def test_selective_review_no_escalation():
    backend = MockBackend(default_response="B")
    executor = Executor(backends={"mock": backend})
    pipeline = SelectiveReviewPipeline(
        executor=executor,
        specialist_selector=lambda d: _make_model("specialist"),
        senior_reviewer=_make_model("senior"),
        analyzer=TaskAnalyzer(),
    )
    result = await pipeline.run("What is 2+2?")
    assert not result.escalated
    assert pipeline.review_count == 0


@pytest.mark.asyncio
async def test_cascade_skips_laborer_specialist_without_duplicate_attempts():
    laborer = _make_model("laborer")
    senior = _make_model("senior")
    calls = []

    async def generate(model, prompt, max_tokens=256, temperature=0.0):
        calls.append((model.name, prompt))
        if model.name == "senior":
            return ExecutionResult(
                text="C",
                confidence=0.9,
                model_used=model.name,
                latency_ms=30,
                token_count=1,
            )

        text = "A" if len(calls) == 1 else "B"
        latency = 10 if len(calls) == 1 else 20
        return ExecutionResult(
            text=text,
            confidence=0.5,
            model_used=model.name,
            latency_ms=latency,
            token_count=1,
        )

    backend = MockBackend()
    backend.generate = generate
    executor = Executor(backends={"mock": backend})
    pipeline = CascadePipeline(
        executor=executor,
        laborer=laborer,
        specialist_selector=lambda d: laborer,
        senior_reviewer=senior,
        analyzer=TaskAnalyzer(),
    )

    result = await pipeline.run("If all bloops are razzies, what follows?")

    assert [name for name, _ in calls] == ["laborer", "laborer", "senior"]
    assert pipeline.specialist_count == 0
    assert pipeline.senior_count == 1
    assert result.model_used == "laborer -> senior"
    assert result.total_latency_ms == 60

    review_prompt = calls[-1][1]
    assert "Attempt 1 (laborer): A" in review_prompt
    assert "Attempt 2 (laborer): B" in review_prompt
    assert "Attempt 3" not in review_prompt
