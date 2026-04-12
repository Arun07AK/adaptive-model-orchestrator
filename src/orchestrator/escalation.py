from __future__ import annotations

from src.models.registry import ModelRegistry
from src.orchestrator.executor import Executor
from src.types import Domain, ExecutionResult


class EscalationStrategy:
    def __init__(
        self,
        executor: Executor,
        registry: ModelRegistry,
        threshold: float = 0.6,
    ) -> None:
        self._executor = executor
        self._registry = registry
        self._threshold = threshold

    def should_escalate(self, confidence: float) -> bool:
        return confidence < self._threshold

    async def escalate(
        self,
        prompt: str,
        original_domain: Domain,
        max_tokens: int = 256,
    ) -> ExecutionResult:
        escalation_model = self._registry.get_escalation_model()
        return await self._executor.execute(
            model=escalation_model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
