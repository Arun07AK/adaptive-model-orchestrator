from __future__ import annotations

from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.aggregator import Aggregator
from src.orchestrator.decomposer import Decomposer
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.executor import Executor
from src.orchestrator.router import Router
from src.types import OrchestratorResult


class OrchestratorPipeline:
    def __init__(
        self,
        analyzer: TaskAnalyzer,
        router: Router,
        executor: Executor,
        escalation: EscalationStrategy,
        aggregator: Aggregator,
        decomposer: Decomposer | None = None,
        enable_escalation: bool = True,
        enable_decomposition: bool = False,
    ) -> None:
        self._analyzer = analyzer
        self._router = router
        self._executor = executor
        self._escalation = escalation
        self._aggregator = aggregator
        self._decomposer = decomposer
        self._enable_escalation = enable_escalation
        self._enable_decomposition = enable_decomposition

    async def run(
        self,
        prompt: str,
        subject_hint: str | None = None,
        max_tokens: int = 256,
    ) -> OrchestratorResult:
        analysis = self._analyzer.classify(prompt, subject_hint=subject_hint)

        if (
            self._enable_decomposition
            and self._decomposer is not None
            and self._decomposer.should_decompose(analysis)
        ):
            subtasks = await self._decomposer.decompose(analysis)
            results = []
            for subtask in subtasks:
                sub_analysis = self._analyzer.classify(subtask)
                decision = self._router.route(sub_analysis)
                result = await self._executor.execute(
                    model=decision.model, prompt=subtask, max_tokens=max_tokens,
                )
                results.append(result)
            return self._aggregator.aggregate(results)

        decision = self._router.route(analysis)

        result = await self._executor.execute(
            model=decision.model, prompt=prompt, max_tokens=max_tokens,
        )

        if self._enable_escalation and self._escalation.should_escalate(result.confidence):
            escalated = await self._escalation.escalate(
                prompt=prompt,
                original_domain=analysis.domain,
                max_tokens=max_tokens,
            )
            return OrchestratorResult(
                text=escalated.text,
                model_used=escalated.model_used,
                escalated=True,
                escalation_model=escalated.model_used,
                total_latency_ms=result.latency_ms + escalated.latency_ms,
                confidence=escalated.confidence,
            )

        return OrchestratorResult(
            text=result.text,
            model_used=result.model_used,
            escalated=False,
            total_latency_ms=result.latency_ms,
            confidence=result.confidence,
        )
