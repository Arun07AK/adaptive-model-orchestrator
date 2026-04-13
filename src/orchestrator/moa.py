from __future__ import annotations

import asyncio
from src.orchestrator.executor import Executor
from src.types import ExecutionResult, ModelConfig, OrchestratorResult

_AGGREGATOR_SYSTEM_PROMPT = (
    "You have been provided with a set of responses from various open-source models "
    "to the latest user query. Your task is to synthesize these responses into a single, "
    "high-quality response. It is crucial to critically evaluate the information provided "
    "in these responses, recognizing that some of it may be biased or incorrect. Your "
    "response should not simply replicate the given answers but should offer a refined, "
    "accurate, and comprehensive reply to the instruction. Ensure your response is "
    "well-structured, coherent, and adheres to the highest standards of accuracy and reliability."
)


class MixtureOfAgents:
    def __init__(
        self,
        executor: Executor,
        proposer_models: list[ModelConfig],
        aggregator_model: ModelConfig,
    ) -> None:
        self._executor = executor
        self._proposers = proposer_models
        self._aggregator = aggregator_model

    async def run(
        self,
        prompt: str,
        subject_hint: str | None = None,
        max_tokens: int = 256,
        proposer_max_tokens: int | None = None,
    ) -> OrchestratorResult:
        p_tokens = proposer_max_tokens or max_tokens

        # Layer 1: All proposers answer independently (sequentially to avoid rate limits)
        proposals: list[ExecutionResult] = []
        for model in self._proposers:
            result = await self._executor.execute(
                model=model, prompt=prompt, max_tokens=p_tokens,
            )
            proposals.append(result)

        # Build aggregator prompt
        agg_prompt = self._build_aggregator_prompt(prompt, proposals)

        # Layer 2: Aggregator synthesizes
        import time
        start = time.perf_counter()
        agg_result = await self._executor.execute(
            model=self._aggregator,
            prompt=agg_prompt,
            max_tokens=max_tokens,
        )
        total_latency = sum(p.latency_ms for p in proposals) + agg_result.latency_ms

        models_used = ", ".join(dict.fromkeys(
            [p.model_used for p in proposals] + [agg_result.model_used]
        ))

        return OrchestratorResult(
            text=agg_result.text,
            model_used=models_used,
            escalated=False,
            total_latency_ms=total_latency,
            confidence=agg_result.confidence,
        )

    def _build_aggregator_prompt(
        self, original_prompt: str, proposals: list[ExecutionResult]
    ) -> str:
        refs = "\n\n".join(
            f"{i+1}. {p.text}" for i, p in enumerate(proposals)
        )
        return (
            f"{_AGGREGATOR_SYSTEM_PROMPT}\n\n"
            f"Responses from models:\n{refs}\n\n"
            f"Original question: {original_prompt}\n\n"
            f"Your synthesized response:"
        )
