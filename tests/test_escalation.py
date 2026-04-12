import pytest
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.executor import Executor
from src.models.registry import ModelRegistry
from src.types import ExecutionResult, Domain
from tests.conftest import MockBackend


@pytest.fixture
def backend():
    return MockBackend(default_response="escalated answer", default_confidence=0.95)


@pytest.fixture
def strategy(backend):
    executor = Executor(backends={"mlx": backend, "groq": backend})
    registry = ModelRegistry()
    return EscalationStrategy(executor=executor, registry=registry, threshold=0.6)


def test_should_escalate_low_confidence(strategy):
    assert strategy.should_escalate(confidence=0.3) is True


def test_should_not_escalate_high_confidence(strategy):
    assert strategy.should_escalate(confidence=0.8) is False


def test_should_escalate_at_boundary(strategy):
    assert strategy.should_escalate(confidence=0.6) is False
    assert strategy.should_escalate(confidence=0.59) is True


@pytest.mark.asyncio
async def test_escalate_calls_stronger_model(strategy):
    result = await strategy.escalate(prompt="Hard question", original_domain=Domain.MATH)
    assert result.text == "escalated answer"
    assert result.model_used == "llama-3.3-70b"


@pytest.mark.asyncio
async def test_escalate_returns_execution_result(strategy):
    result = await strategy.escalate(prompt="Hard question", original_domain=Domain.MATH)
    assert isinstance(result, ExecutionResult)
