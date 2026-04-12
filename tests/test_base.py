import pytest
from tests.conftest import MockBackend
from src.types import ModelConfig, Domain, CostTier


@pytest.fixture
def mock_backend():
    return MockBackend(default_response="42", default_confidence=0.95)


@pytest.fixture
def math_model():
    return ModelConfig(
        name="test-math-7b",
        provider="mock",
        domain=Domain.MATH,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="test-math",
        ram_gb=5.0,
    )


@pytest.mark.asyncio
async def test_generate_returns_execution_result(mock_backend, math_model):
    result = await mock_backend.generate(
        model=math_model,
        prompt="What is 6 * 7?",
        max_tokens=100,
    )
    assert result.text == "42"
    assert result.confidence == 0.95
    assert result.model_used == "test-math-7b"


@pytest.mark.asyncio
async def test_generate_tracks_call_count(mock_backend, math_model):
    await mock_backend.generate(model=math_model, prompt="q1", max_tokens=10)
    await mock_backend.generate(model=math_model, prompt="q2", max_tokens=10)
    assert mock_backend.call_count == 2


@pytest.mark.asyncio
async def test_generate_records_prompts(mock_backend, math_model):
    await mock_backend.generate(model=math_model, prompt="What is pi?", max_tokens=10)
    assert mock_backend.last_prompt == "What is pi?"
