import pytest

from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.cascade import (
    CrossModelConsistencyScorer,
    SelfConsistencyScorer,
    SelectiveReviewPipeline,
    _normalize_answer,
)
from src.orchestrator.executor import Executor
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


def test_normalize_answer_uses_final_letter_not_earlier_distractor():
    assert _normalize_answer("A is tempting, but the final answer is D.") == "D"


def test_normalize_answer_number():
    assert _normalize_answer("The answer is 42") == "42"
    assert _normalize_answer("#### 72") == "72"


def test_normalize_answer_uses_final_number_not_reasoning_prefix():
    assert _normalize_answer("There are 5 groups of 10, so #### 50") == "50"


def test_normalize_answer_prefers_numeric_marker_over_letter_fallback():
    assert _normalize_answer("A and B each have 5, so #### 10") == "10"


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
async def test_cross_model_consistency_disagrees_on_different_final_answers():
    backend = MockBackend()
    backend.set_response("llama", "q?", "A is tempting, but final answer is D.", 0.9)
    backend.set_response("qwen", "q?", "A is tempting, but final answer is C.", 0.9)
    executor = Executor(backends={"mock": backend})
    scorer = CrossModelConsistencyScorer(executor)

    attempts, consistent = await scorer.score(
        _make_model("llama"), _make_model("qwen"), "q?", max_tokens=10
    )

    assert len(attempts) == 2
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
