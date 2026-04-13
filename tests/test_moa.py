import pytest
from src.orchestrator.moa import MixtureOfAgents
from src.orchestrator.executor import Executor
from src.types import ModelConfig, Domain, CostTier, OrchestratorResult
from tests.conftest import MockBackend


def _make_model(name: str) -> ModelConfig:
    return ModelConfig(
        name=name, provider="mock", domain=Domain.GENERAL,
        size_b=7.0, cost_tier=CostTier.FREE_API, model_id=name,
    )


@pytest.fixture
def moa():
    backend = MockBackend(default_response="test answer", default_confidence=0.9)
    backend.set_response("proposer-1", "", "Answer A", 0.8)
    backend.set_response("proposer-2", "", "Answer B", 0.7)
    backend.set_response("proposer-3", "", "Answer C", 0.9)
    backend.set_response("aggregator", "", "Final synthesized answer", 0.95)
    executor = Executor(backends={"mock": backend})
    return MixtureOfAgents(
        executor=executor,
        proposer_models=[_make_model("proposer-1"), _make_model("proposer-2"), _make_model("proposer-3")],
        aggregator_model=_make_model("aggregator"),
    )


@pytest.mark.asyncio
async def test_moa_returns_orchestrator_result(moa):
    result = await moa.run("What is 2+2?")
    assert isinstance(result, OrchestratorResult)


@pytest.mark.asyncio
async def test_moa_uses_all_proposers(moa):
    result = await moa.run("What is 2+2?")
    assert "proposer-1" in result.model_used
    assert "proposer-2" in result.model_used
    assert "proposer-3" in result.model_used
    assert "aggregator" in result.model_used


@pytest.mark.asyncio
async def test_moa_aggregator_sees_proposals(moa):
    backend = MockBackend()
    executor = Executor(backends={"mock": backend})
    moa_inst = MixtureOfAgents(
        executor=executor,
        proposer_models=[_make_model("p1"), _make_model("p2")],
        aggregator_model=_make_model("agg"),
    )
    await moa_inst.run("test question")
    # Aggregator's prompt should contain "Responses from models"
    assert "Responses from models" in backend.last_prompt
    assert "test question" in backend.last_prompt


@pytest.mark.asyncio
async def test_moa_tracks_total_latency(moa):
    result = await moa.run("What is 2+2?")
    # 3 proposers + 1 aggregator = 4 calls, each 10ms
    assert result.total_latency_ms >= 40.0
