import pytest
from src.orchestrator.cascade import (
    CrossModelPipeline,
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


def test_normalize_answer_prefers_final_numeric_answer():
    assert _normalize_answer(
        "There are 16 items at first, so the answer is #### 12",
    ) == "12"
    assert _normalize_answer("Start with 16.\nTherefore, \\boxed{10}") == "10"


def test_normalize_answer_prefers_explicit_choice_over_option_echo():
    response = "A. methane\nB. oxygen\nC. nitrogen\nAfter checking, final answer: B"
    assert _normalize_answer(response) == "B"


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
async def test_cross_model_escalates_when_final_numbers_differ():
    backend = MockBackend(default_response="#### 99")
    backend.set_response(
        "model-a",
        "apples",
        "We both start with 16 apples, then compute carefully. #### 12",
        0.8,
    )
    backend.set_response(
        "model-b",
        "apples",
        "We both start with 16 apples, then compute carefully. #### 10",
        0.8,
    )
    executor = Executor(backends={"mock": backend})
    pipeline = CrossModelPipeline(
        executor=executor,
        model_a=_make_model("model-a"),
        model_b=_make_model("model-b"),
        senior_reviewer=_make_model("senior"),
        analyzer=TaskAnalyzer(),
    )

    result = await pipeline.run("apples", max_tokens=10)

    assert result.escalated
    assert pipeline.review_count == 1
    assert backend.call_count == 3
