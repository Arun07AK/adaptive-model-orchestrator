from __future__ import annotations

from typing import Protocol

from src.types import ExecutionResult, ModelConfig


class InferenceBackend(Protocol):
    async def generate(
        self,
        model: ModelConfig,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ExecutionResult: ...
