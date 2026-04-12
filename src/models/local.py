from __future__ import annotations

import asyncio
import time

from src.types import ExecutionResult, ModelConfig

try:
    from mlx_lm import load as _mlx_load
    from mlx_lm import generate as _mlx_generate
except ImportError:
    _mlx_load = None
    _mlx_generate = None


def mlx_lm_load(model_id: str) -> tuple:
    if _mlx_load is None:
        raise ImportError("mlx-lm not installed")
    return _mlx_load(model_id)


def mlx_lm_generate(model, tokenizer, prompt: str, max_tokens: int, temp: float) -> str:
    if _mlx_generate is None:
        raise ImportError("mlx-lm not installed")
    return _mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temp)


class MLXBackend:
    def __init__(self) -> None:
        self._loaded_models: dict[str, tuple] = {}

    def _ensure_loaded(self, model: ModelConfig) -> tuple:
        if model.model_id not in self._loaded_models:
            loaded_model, tokenizer = mlx_lm_load(model.model_id)
            self._loaded_models[model.model_id] = (loaded_model, tokenizer)
        return self._loaded_models[model.model_id]

    async def generate(
        self,
        model: ModelConfig,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ExecutionResult:
        mlx_model, tokenizer = self._ensure_loaded(model)

        start = time.perf_counter()
        text = await asyncio.to_thread(
            mlx_lm_generate, mlx_model, tokenizer, prompt, max_tokens, temperature
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExecutionResult(
            text=text,
            confidence=0.0,
            model_used=model.name,
            latency_ms=elapsed_ms,
            token_count=len(text.split()),
        )
