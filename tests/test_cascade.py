import pytest
from src.orchestrator.cascade import (
    CascadePipeline,
    SelectiveReviewPipeline,
    SelfConsistencyScorer,
    _normalize_answer,
)
from src.orchestrator.executor import Executor
from src.orchestrator.analyzer import TaskAnalyzer
from src.types import CostTier, Domain, ModelConfig
from tests.conftest import MockBackend


def _make_model(name, domain=Domain.GENERAL, size=7.0):
    return ModelConfig(
        name=name, provider="mock", domain=domain, size_b=size,
        cost_tier=CostTier.FREE_API, model_id=name,
    )


def test_normalize_answer_letter():
    assert _normalize_answer("The answer is B.") == "B"
    assert _normalize_answer("<think>reasoning</think>A") == "A"
    assert _normalize_answer("Option A is tempting. Final answer: B.") == "B"


def test_normalize_answer_number():
    assert _normalize_answer("The answer is 42") == "42"
    assert _normalize_answer("#### 72") == "72"
    assert _normalize_answer("We computed 100 first. #### 72") == "72"


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
        return ExecutionResult(
            text=text,
            confidence=0.5,
            model_used=model.name,
            latency_ms=10,
            token_count=1,
        )

    backend.generate = varied
    executor = Executor(backends={"mock": backend})
    scorer = SelfConsistencyScorer(executor)
    _attempts, consistent = await scorer.score(_make_model("m1"), "q?", max_tokens=10)
    assert consistent is False


@pytest.mark.asyncio
async def test_self_consistency_compares_final_answers():
    backend = MockBackend()
    call = [0]

    async def varied(model, prompt, max_tokens=256, temperature=0.0):
        from src.types import ExecutionResult

        call[0] += 1
        text = (
            "Option A is plausible. Final answer: B."
            if call[0] == 1
            else "Option A is plausible. Final answer: C."
        )
        return ExecutionResult(
            text=text,
            confidence=0.5,
            model_used=model.name,
            latency_ms=10,
            token_count=1,
        )

    backend.generate = varied
    executor = Executor(backends={"mock": backend})
    scorer = SelfConsistencyScorer(executor)
    _attempts, consistent = await scorer.score(_make_model("m1"), "q?", max_tokens=10)
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
async def test_cascade_skips_duplicate_attempts_when_specialist_is_laborer():
    backend = MockBackend(default_response="reviewed answer", latency_ms=10)
    call = [0]

    async def varied(model, prompt, max_tokens=256, temperature=0.0):
        from src.types import ExecutionResult

        call[0] += 1
        if model.name == "laborer":
            text = "Final answer: A" if call[0] == 1 else "Final answer: B"
        else:
            text = "Final answer: C"
        backend.last_prompt = prompt
        return ExecutionResult(
            text=text,
            confidence=0.5,
            model_used=model.name,
            latency_ms=10,
            token_count=1,
        )

    backend.generate = varied
    executor = Executor(backends={"mock": backend})
    laborer = _make_model("laborer", size=8.0)
    pipeline = CascadePipeline(
        executor=executor,
        laborer=laborer,
        specialist_selector=lambda d: laborer,
        senior_reviewer=_make_model("senior", size=70.0),
        analyzer=TaskAnalyzer(),
    )

    result = await pipeline.run("What follows?", max_tokens=10)

    assert result.escalated
    assert result.model_used == "laborer -> senior"
    assert result.total_latency_ms == 30
    assert backend.last_prompt is not None
    assert backend.last_prompt.count("Attempt ") == 2
    assert pipeline.specialist_count == 0
    assert pipeline.senior_count == 1
