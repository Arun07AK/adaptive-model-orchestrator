import pytest
from unittest.mock import MagicMock, patch
from src.models.local import MLXBackend
from src.types import ModelConfig, Domain, CostTier


@pytest.fixture
def math_model():
    return ModelConfig(
        name="deepseek-r1-distill-qwen-7b",
        provider="mlx",
        domain=Domain.MATH,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        ram_gb=5.0,
    )


def test_mlx_backend_init():
    backend = MLXBackend()
    assert backend._loaded_models == {}


@pytest.mark.asyncio
async def test_generate_calls_mlx_lm(math_model):
    backend = MLXBackend()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("src.models.local.mlx_lm_load", return_value=(mock_model, mock_tokenizer)):
        with patch("src.models.local.mlx_lm_generate", return_value="42"):
            result = await backend.generate(
                model=math_model,
                prompt="What is 6*7?",
                max_tokens=100,
            )

    assert result.text == "42"
    assert result.model_used == "deepseek-r1-distill-qwen-7b"
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_generate_caches_loaded_model(math_model):
    backend = MLXBackend()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("src.models.local.mlx_lm_load", return_value=(mock_model, mock_tokenizer)) as load_mock:
        with patch("src.models.local.mlx_lm_generate", return_value="42"):
            await backend.generate(model=math_model, prompt="q1", max_tokens=10)
            await backend.generate(model=math_model, prompt="q2", max_tokens=10)

    load_mock.assert_called_once()
