from __future__ import annotations

from src.models.base import InferenceBackend
from src.models.registry import ModelRegistry
from src.types import Complexity, Domain, ModelConfig, TaskAnalysis

_DECOMPOSE_PROMPT = (
    "Decompose the following task into independent subtasks. "
    "Return each subtask on its own numbered line (e.g., '1. ...'). "
    "Only list the subtasks, no other text.\n\n"
    "Task: {task}"
)


class Decomposer:
    def __init__(
        self,
        backend: InferenceBackend,
        decomposer_model_name: str = "qwen2.5-7b",
        registry: ModelRegistry | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry or ModelRegistry()
        self._model = self._registry.get_by_name(decomposer_model_name)
        if self._model is None:
            self._model = self._registry.get_cheapest(Domain.GENERAL)

    def should_decompose(self, analysis: TaskAnalysis) -> bool:
        return analysis.complexity == Complexity.COMPLEX

    async def decompose(self, analysis: TaskAnalysis) -> list[str]:
        assert self._model is not None
        prompt = _DECOMPOSE_PROMPT.format(task=analysis.text)
        result = await self._backend.generate(
            model=self._model,
            prompt=prompt,
            max_tokens=512,
        )
        return self._parse_subtasks(result.text)

    @staticmethod
    def _parse_subtasks(text: str) -> list[str]:
        subtasks: list[str] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix_end in (". ", ") ", "- "):
                idx = line.find(prefix_end)
                if idx != -1 and idx < 4:
                    line = line[idx + len(prefix_end):]
                    break
            if line:
                subtasks.append(line)
        return subtasks
