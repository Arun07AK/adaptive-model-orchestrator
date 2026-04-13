from __future__ import annotations

import asyncio
import time

from src.types import ExecutionResult, ModelConfig

try:
    from litellm import acompletion
except ImportError:
    acompletion = None

_MAX_RETRIES = 5


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
        for attempt in range(_MAX_RETRIES):
            try:
                response = await acompletion(
                    model=litellm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "limit" in err or "429" in str(e) or "too many" in err:
                    wait = 3 * (attempt + 1)  # 3s, 6s, 9s, 12s, 15s
                    await asyncio.sleep(wait)
                    if attempt == _MAX_RETRIES - 1:
                        raise
                else:
                    raise
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
