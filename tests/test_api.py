import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.models.api import LiteLLMBackend
from src.types import ModelConfig, Domain, CostTier


@pytest.fixture
def groq_model():
    return ModelConfig(
        name="llama-3.3-70b",
        provider="groq",
        domain=Domain.GENERAL,
        size_b=70.0,
        cost_tier=CostTier.FREE_API,
        model_id="llama-3.3-70b-versatile",
    )


def test_litellm_backend_formats_model_id():
    backend = LiteLLMBackend()
    assert backend._format_model_id("groq", "llama-3.3-70b-versatile") == "groq/llama-3.3-70b-versatile"
    assert backend._format_model_id("together", "meta-llama/Llama-3.3-70B") == "together_ai/meta-llama/Llama-3.3-70B"


@pytest.mark.asyncio
async def test_generate_calls_litellm(groq_model):
    backend = LiteLLMBackend()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Paris is the capital of France."
    mock_response.usage.completion_tokens = 7

    with patch("src.models.api.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await backend.generate(
            model=groq_model,
            prompt="What is the capital of France?",
            max_tokens=100,
        )

    assert result.text == "Paris is the capital of France."
    assert result.model_used == "llama-3.3-70b"
    assert result.token_count == 7
    assert result.latency_ms > 0
