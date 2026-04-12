from __future__ import annotations

from src.types import ExecutionResult, OrchestratorResult


class Aggregator:
    def aggregate(self, results: list[ExecutionResult]) -> OrchestratorResult:
        if not results:
            raise ValueError("No results to aggregate")

        if len(results) == 1:
            r = results[0]
            return OrchestratorResult(
                text=r.text,
                model_used=r.model_used,
                escalated=False,
                total_latency_ms=r.latency_ms,
                confidence=r.confidence,
            )

        combined_text = "\n\n".join(r.text for r in results)
        models_used = ", ".join(dict.fromkeys(r.model_used for r in results))
        total_latency = sum(r.latency_ms for r in results)
        min_confidence = min(r.confidence for r in results)

        return OrchestratorResult(
            text=combined_text,
            model_used=models_used,
            escalated=False,
            total_latency_ms=total_latency,
            confidence=min_confidence,
        )
