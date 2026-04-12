"""Integration test: run a small batch of questions through all 3 pipeline configs."""
import pytest
from src.orchestrator.pipeline import OrchestratorPipeline
from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.router import Router
from src.orchestrator.executor import Executor
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.aggregator import Aggregator
from src.models.registry import ModelRegistry
from src.types import Domain, OrchestratorResult, RoutingDecision
from tests.conftest import MockBackend


def _build_smart_backend() -> MockBackend:
    backend = MockBackend(default_response="general answer", default_confidence=0.7)
    backend.set_response("deepseek-r1-distill-qwen-7b", "integral", "x^3/3 + C", 0.95)
    backend.set_response("deepseek-r1-distill-qwen-7b", "derivative", "2x + 3", 0.93)
    backend.set_response("deepseek-coder-v2-16b", "function", "def sort(arr): ...", 0.90)
    backend.set_response("deepseek-coder-v2-16b", "Python", "def solve(): ...", 0.88)
    backend.set_response("qwen2.5-7b", "capital", "Paris", 0.92)
    backend.set_response("llama-3.3-70b", "", "high-quality escalated answer", 0.96)
    return backend


QUESTIONS = [
    ("What is the integral of x^2?", Domain.MATH),
    ("What is the derivative of x^2 + 3x?", Domain.MATH),
    ("Write a Python function to sort a list", Domain.CODE),
    ("Implement a binary search function", Domain.CODE),
    ("What is the capital of France?", Domain.GENERAL),
    ("Who wrote Romeo and Juliet?", Domain.GENERAL),
]


def _build_pipeline(enable_escalation: bool, force_single: bool = False) -> OrchestratorPipeline:
    backend = _build_smart_backend()
    registry = ModelRegistry()
    executor = Executor(backends={"mlx": backend, "groq": backend})
    escalation = EscalationStrategy(executor=executor, registry=registry, threshold=0.6)

    if force_single:
        class SingleRouter(Router):
            def route(self, analysis, prefer_stronger=False):
                model = self._registry.get_cheapest(Domain.GENERAL)
                return RoutingDecision(model=model, reason="single-model baseline")
        router = SingleRouter(registry=registry)
    else:
        router = Router(registry=registry)

    return OrchestratorPipeline(
        analyzer=TaskAnalyzer(),
        router=router,
        executor=executor,
        escalation=escalation,
        aggregator=Aggregator(),
        enable_escalation=enable_escalation,
    )


@pytest.mark.asyncio
async def test_single_model_config():
    pipeline = _build_pipeline(enable_escalation=False, force_single=True)
    for question, _ in QUESTIONS:
        result = await pipeline.run(question)
        assert isinstance(result, OrchestratorResult)
        assert result.model_used == "qwen2.5-7b"
        assert not result.escalated


@pytest.mark.asyncio
async def test_orchestrated_config():
    pipeline = _build_pipeline(enable_escalation=False)
    results = {}
    for question, expected_domain in QUESTIONS:
        result = await pipeline.run(question)
        results[question] = result

    for result in results.values():
        assert not result.escalated


@pytest.mark.asyncio
async def test_escalation_config():
    pipeline = _build_pipeline(enable_escalation=True)
    results = []
    for question, _ in QUESTIONS:
        result = await pipeline.run(question)
        results.append(result)

    non_escalated = [r for r in results if not r.escalated]
    assert len(non_escalated) >= 1


@pytest.mark.asyncio
async def test_orchestrated_routes_to_different_models():
    pipeline = _build_pipeline(enable_escalation=False)
    models_used = set()
    for question, _ in QUESTIONS:
        result = await pipeline.run(question)
        models_used.add(result.model_used)

    assert len(models_used) >= 2, f"Only used: {models_used}"
