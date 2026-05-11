import pytest
from src.orchestrator.cascade import (
    CrossModelConsistencyScorer,
    SelectiveReviewPipeline,
    SelfConsistencyScorer,
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


def test_normalize_answer_prefers_final_mcq_answer():
    assert _normalize_answer("A is tempting, but the final answer is B.") == "B"
    assert _normalize_answer("A) eliminate this option.\nTherefore choose D.") == "D"


def test_normalize_answer_prefers_final_numeric_answer():
    assert _normalize_answer("There are 3 groups, so 3 * 24 = 72.\n#### 72") == "72"
    assert _normalize_answer("First compute 10. The answer is 8.") == "8"
    assert _normalize_answer("A total of 12 widgets remain.") == "12"
    assert _normalize_answer("Trial 1 gives 5, but corrected total is \\boxed{12}.") == "12"


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
async def test_self_consistency_uses_final_numeric_answer():
    backend = MockBackend()
    call = [0]

    async def varied(model, prompt, max_tokens=256, temperature=0.0):
        from src.types import ExecutionResult

        call[0] += 1
        final = "72" if call[0] == 1 else "71"
        text = f"A total is computed from the same setup.\n#### {final}"
        return ExecutionResult(
            text=text,
            confidence=0.5,
            model_used=model.name,
            latency_ms=10,
            token_count=8,
        )

    backend.generate = varied
    executor = Executor(backends={"mock": backend})
    scorer = SelfConsistencyScorer(executor)
    attempts, consistent = await scorer.score(_make_model("m1"), "q?", max_tokens=10)

    assert [a.text for a in attempts] == [
        "A total is computed from the same setup.\n#### 72",
        "A total is computed from the same setup.\n#### 71",
    ]
    assert consistent is False


@pytest.mark.asyncio
async def test_cross_model_consistency_uses_final_mcq_answer():
    backend = MockBackend()

    async def by_model(model, prompt, max_tokens=256, temperature=0.0):
        from src.types import ExecutionResult

        text = (
            "A is an attractive distractor. Final answer: B"
            if model.name == "model-a"
            else "A is an attractive distractor. Final answer: C"
        )
        return ExecutionResult(
            text=text,
            confidence=0.5,
            model_used=model.name,
            latency_ms=10,
            token_count=8,
        )

    backend.generate = by_model
    executor = Executor(backends={"mock": backend})
    scorer = CrossModelConsistencyScorer(executor)
    _, consistent = await scorer.score(
        _make_model("model-a"), _make_model("model-b"), "q?", max_tokens=10
    )

    assert consistent is False
