from __future__ import annotations

from src.models.registry import ModelRegistry
from src.types import RoutingDecision, TaskAnalysis

_LOW_CONFIDENCE_THRESHOLD = 0.5


class Router:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def route(
        self,
        analysis: TaskAnalysis,
        prefer_stronger: bool = False,
    ) -> RoutingDecision:
        domain = analysis.domain

        if prefer_stronger or analysis.confidence < _LOW_CONFIDENCE_THRESHOLD:
            model = self._registry.get_strongest(domain)
            reason = (
                f"Low classification confidence ({analysis.confidence:.2f}), "
                f"using strongest {domain.value} model: {model.name}"
            )
        else:
            model = self._registry.get_cheapest(domain)
            reason = (
                f"Routed to cheapest {domain.value} specialist: {model.name} "
                f"(confidence: {analysis.confidence:.2f})"
            )

        return RoutingDecision(model=model, reason=reason)
