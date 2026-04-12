from __future__ import annotations

import asyncio
from typing import Any

from src.orchestrator.pipeline import OrchestratorPipeline

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False

    class LM:
        pass

    class Instance:
        pass


class OrchestratorLM(LM):
    def __init__(self, pipeline: OrchestratorPipeline, **kwargs: Any) -> None:
        if _HAS_LM_EVAL:
            super().__init__(**kwargs)
        self._pipeline = pipeline
        self._loop = asyncio.new_event_loop()

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        for request in requests:
            prompt = request.args[0]
            subject_hint = None
            if hasattr(request, "doc") and isinstance(request.doc, dict):
                subject_hint = request.doc.get("subject")

            orchestrator_result = self._loop.run_until_complete(
                self._pipeline.run(prompt, subject_hint=subject_hint)
            )
            results.append(orchestrator_result.text)
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args
            full_prompt = context + continuation

            orchestrator_result = self._loop.run_until_complete(
                self._pipeline.run(full_prompt)
            )

            generated = orchestrator_result.text.strip().lower()
            expected = continuation.strip().lower()
            is_match = generated.startswith(expected) or expected in generated

            log_likelihood = 0.0 if is_match else -10.0
            results.append((log_likelihood, is_match))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return self.loglikelihood(requests)
