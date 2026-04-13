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


def mlx_lm_generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    if _mlx_generate is None:
        raise ImportError("mlx-lm not installed")
    return _mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


class MLXBackend:
    def __init__(self, max_cached: int = 1) -> None:
        self._loaded_models: dict[str, tuple] = {}
        self._max_cached = max_cached
        self._load_order: list[str] = []

    def _ensure_loaded(self, model: ModelConfig) -> tuple:
        if model.model_id not in self._loaded_models:
            # Evict oldest models if at capacity
            while len(self._loaded_models) >= self._max_cached:
                evict_id = self._load_order.pop(0)
                del self._loaded_models[evict_id]
                import gc; gc.collect()

            loaded_model, tokenizer = mlx_lm_load(model.model_id)
            self._loaded_models[model.model_id] = (loaded_model, tokenizer)
            self._load_order.append(model.model_id)
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
            mlx_lm_generate, mlx_model, tokenizer, prompt, max_tokens
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExecutionResult(
            text=text,
            confidence=0.0,
            model_used=model.name,
            latency_ms=elapsed_ms,
            token_count=len(text.split()),
        )
