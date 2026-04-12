from __future__ import annotations

import time

from src.types import ExecutionResult, ModelConfig

try:
    from litellm import acompletion
except ImportError:
    acompletion = None


_PROVIDER_PREFIX = {
    "groq": "groq",
    "together": "together_ai",
}


class LiteLLMBackend:
    def _format_model_id(self, provider: str, model_id: str) -> str:
        prefix = _PROVIDER_PREFIX.get(provider, provider)
        return f"{prefix}/{model_id}"

    async def generate(
        self,
        model: ModelConfig,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ExecutionResult:
        if acompletion is None:
            raise ImportError("litellm not installed")

        litellm_model = self._format_model_id(model.provider, model.model_id)

        start = time.perf_counter()
        response = await acompletion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""
        token_count = response.usage.completion_tokens if response.usage else len(text.split())

        return ExecutionResult(
            text=text,
            confidence=0.0,
            model_used=model.name,
            latency_ms=elapsed_ms,
            token_count=token_count,
        )
