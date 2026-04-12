from __future__ import annotations

import pytest

from src.types import (
    CostTier, Domain, ExecutionResult, ModelConfig,
)


class MockBackend:
    def __init__(
        self,
        default_response: str = "mock response",
        default_confidence: float = 0.9,
        latency_ms: float = 10.0,
    ) -> None:
        self.default_response = default_response
        self.default_confidence = default_confidence
        self.latency_ms = latency_ms
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_model: ModelConfig | None = None
        self._responses: dict[tuple[str, str], tuple[str, float]] = {}

    def set_response(
        self, model_name: str, prompt_contains: str, response: str, confidence: float
    ) -> None:
        self._responses[(model_name, prompt_contains)] = (response, confidence)

    async def generate(
        self,
        model: ModelConfig,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ExecutionResult:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_model = model

        for (m_name, p_sub), (resp, conf) in self._responses.items():
            if model.name == m_name and p_sub in prompt:
                return ExecutionResult(
                    text=resp,
                    confidence=conf,
                    model_used=model.name,
                    latency_ms=self.latency_ms,
                    token_count=len(resp.split()),
                )

        return ExecutionResult(
            text=self.default_response,
            confidence=self.default_confidence,
            model_used=model.name,
            latency_ms=self.latency_ms,
            token_count=len(self.default_response.split()),
        )


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def math_model_local():
    return ModelConfig(
        name="deepseek-r1-distill-qwen-7b",
        provider="mlx",
        domain=Domain.MATH,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        ram_gb=5.0,
    )


@pytest.fixture
def code_model_local():
    return ModelConfig(
        name="deepseek-coder-v2-16b",
        provider="mlx",
        domain=Domain.CODE,
        size_b=16.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
        ram_gb=10.0,
    )


@pytest.fixture
def general_model_local():
    return ModelConfig(
        name="qwen2.5-7b",
        provider="mlx",
        domain=Domain.GENERAL,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        ram_gb=5.0,
    )


@pytest.fixture
def escalation_model():
    return ModelConfig(
        name="llama-3.3-70b",
        provider="groq",
        domain=Domain.GENERAL,
        size_b=70.0,
        cost_tier=CostTier.FREE_API,
        model_id="llama-3.3-70b-versatile",
    )
