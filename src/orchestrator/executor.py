from __future__ import annotations

import math

from src.models.base import InferenceBackend
from src.types import ExecutionResult, ModelConfig


class Executor:
    def __init__(self, backends: dict[str, InferenceBackend]) -> None:
        self._backends = backends

    async def execute(
        self,
        model: ModelConfig,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ExecutionResult:
        backend = self._backends.get(model.provider)
        if backend is None:
            raise ValueError(f"No backend for provider '{model.provider}'")

        return await backend.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @staticmethod
    def compute_mcq_confidence(
        answer_token: str,
        log_probs: dict[str, float],
    ) -> float:
        if answer_token not in log_probs:
            return 0.0

        max_lp = max(log_probs.values())
        exp_sum = sum(math.exp(lp - max_lp) for lp in log_probs.values())
        chosen_exp = math.exp(log_probs[answer_token] - max_lp)

        return chosen_exp / exp_sum
