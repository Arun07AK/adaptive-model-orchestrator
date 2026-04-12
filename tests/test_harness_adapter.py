import pytest
from src.benchmarks.harness_adapter import OrchestratorLM
from src.orchestrator.pipeline import OrchestratorPipeline
from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.router import Router
from src.orchestrator.executor import Executor
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.aggregator import Aggregator
from src.models.registry import ModelRegistry
from tests.conftest import MockBackend


@pytest.fixture
def orchestrator_lm():
    backend = MockBackend(default_response="A", default_confidence=0.9)
    registry = ModelRegistry()
    executor = Executor(backends={"mlx": backend, "groq": backend})
    pipeline = OrchestratorPipeline(
        analyzer=TaskAnalyzer(),
        router=Router(registry=registry),
        executor=executor,
        escalation=EscalationStrategy(executor=executor, registry=registry),
        aggregator=Aggregator(),
    )
    return OrchestratorLM(pipeline=pipeline)


def test_orchestrator_lm_has_generate_method(orchestrator_lm):
    assert hasattr(orchestrator_lm, "generate_until")


def test_orchestrator_lm_has_loglikelihood_method(orchestrator_lm):
    assert hasattr(orchestrator_lm, "loglikelihood")
