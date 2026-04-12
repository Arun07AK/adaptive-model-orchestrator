import pytest
from src.orchestrator.executor import Executor
from src.types import ModelConfig, Domain, CostTier
from tests.conftest import MockBackend


@pytest.fixture
def backend():
    return MockBackend(default_response="42", default_confidence=0.95)


@pytest.fixture
def executor(backend):
    return Executor(backends={"mlx": backend, "groq": backend})


@pytest.fixture
def math_model():
    return ModelConfig(name="deepseek-r1-distill-qwen-7b", provider="mlx", domain=Domain.MATH, size_b=7.0, cost_tier=CostTier.LOCAL, model_id="test", ram_gb=5.0)


@pytest.mark.asyncio
async def test_execute_returns_result(executor, math_model):
    result = await executor.execute(model=math_model, prompt="What is 6*7?")
    assert result.text == "42"
    assert result.model_used == "deepseek-r1-distill-qwen-7b"


@pytest.mark.asyncio
async def test_execute_selects_backend_by_provider(executor, math_model):
    backend_mlx = MockBackend(default_response="from_mlx")
    backend_groq = MockBackend(default_response="from_groq")
    executor = Executor(backends={"mlx": backend_mlx, "groq": backend_groq})
    result = await executor.execute(model=math_model, prompt="test")
    assert result.text == "from_mlx"
    assert backend_mlx.call_count == 1
    assert backend_groq.call_count == 0


@pytest.mark.asyncio
async def test_execute_unknown_provider_raises(executor):
    bad_model = ModelConfig(name="test", provider="unknown", domain=Domain.GENERAL, size_b=1.0, cost_tier=CostTier.LOCAL, model_id="x")
    with pytest.raises(ValueError, match="No backend for provider 'unknown'"):
        await executor.execute(model=bad_model, prompt="test")


def test_compute_mcq_confidence():
    executor = Executor(backends={})
    confidence = executor.compute_mcq_confidence(
        answer_token="A",
        log_probs={"A": -0.1, "B": -2.3, "C": -3.5, "D": -4.0},
    )
    assert 0.85 < confidence < 0.87


def test_compute_mcq_confidence_low():
    executor = Executor(backends={})
    confidence = executor.compute_mcq_confidence(
        answer_token="A",
        log_probs={"A": -1.4, "B": -1.4, "C": -1.4, "D": -1.4},
    )
    assert confidence < 0.3
