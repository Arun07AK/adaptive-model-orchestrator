import pytest
from src.orchestrator.cascade import (
    CrossModelPipeline,
    SelfConsistencyScorer,
    SelectiveReviewPipeline,
    _normalize_answer,
)
from src.orchestrator.executor import Executor
from src.orchestrator.analyzer import TaskAnalyzer
from src.types import ModelConfig, Domain, CostTier
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
    assert _normalize_answer("First use 3 groups. #### 72") == "72"
    assert _normalize_answer("The answer is 72.0") == "72"


def test_normalize_answer_prefers_final_letter():
    assert _normalize_answer("A is tempting, and B is plausible. Final answer: C") == "C"


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
async def test_cross_model_escalates_when_final_numbers_disagree():
    backend = MockBackend()
    backend.set_response(
        "model-a",
        "math?",
        "We use 3 groups in the setup. Final answer: 12",
        0.9,
    )
    backend.set_response(
        "model-b",
        "math?",
        "We use 3 groups in the setup. Final answer: 15",
        0.9,
    )
    backend.set_response("senior", "Previous attempts", "Final answer: 15", 0.9)

    executor = Executor(backends={"mock": backend})
    pipeline = CrossModelPipeline(
        executor=executor,
        model_a=_make_model("model-a"),
        model_b=_make_model("model-b"),
        senior_reviewer=_make_model("senior"),
        analyzer=TaskAnalyzer(),
    )

    result = await pipeline.run("math?", max_tokens=10)

    assert result.escalated
    assert result.escalation_model == "senior"
    assert result.text == "Final answer: 15"
    assert pipeline.review_count == 1
