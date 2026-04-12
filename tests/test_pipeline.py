import pytest
from src.orchestrator.pipeline import OrchestratorPipeline
from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.router import Router
from src.orchestrator.executor import Executor
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.aggregator import Aggregator
from src.models.registry import ModelRegistry
from src.types import OrchestratorResult
from tests.conftest import MockBackend


@pytest.fixture
def pipeline():
    backend = MockBackend(default_response="test answer", default_confidence=0.85)
    registry = ModelRegistry()
    executor = Executor(backends={"mlx": backend, "groq": backend})
    return OrchestratorPipeline(
        analyzer=TaskAnalyzer(),
        router=Router(registry=registry),
        executor=executor,
        escalation=EscalationStrategy(executor=executor, registry=registry, threshold=0.6),
        aggregator=Aggregator(),
    )


@pytest.mark.asyncio
async def test_pipeline_math_question(pipeline):
    result = await pipeline.run("What is the integral of x^2?")
    assert isinstance(result, OrchestratorResult)
    assert result.text == "test answer"
    assert not result.escalated


@pytest.mark.asyncio
async def test_pipeline_code_question(pipeline):
    result = await pipeline.run("Write a Python function to sort a list")
    assert isinstance(result, OrchestratorResult)


@pytest.mark.asyncio
async def test_pipeline_general_question(pipeline):
    result = await pipeline.run("What is the capital of France?")
    assert isinstance(result, OrchestratorResult)


@pytest.mark.asyncio
async def test_pipeline_escalation_on_low_confidence():
    backend = MockBackend(default_response="escalated answer", default_confidence=0.3)
    registry = ModelRegistry()
    executor = Executor(backends={"mlx": backend, "groq": backend})
    pipeline = OrchestratorPipeline(
        analyzer=TaskAnalyzer(),
        router=Router(registry=registry),
        executor=executor,
        escalation=EscalationStrategy(executor=executor, registry=registry, threshold=0.6),
        aggregator=Aggregator(),
        enable_escalation=True,
    )
    result = await pipeline.run("What is 2+2?")
    assert result.escalated


@pytest.mark.asyncio
async def test_pipeline_no_escalation_when_disabled():
    backend = MockBackend(default_response="answer", default_confidence=0.3)
    registry = ModelRegistry()
    executor = Executor(backends={"mlx": backend, "groq": backend})
    pipeline = OrchestratorPipeline(
        analyzer=TaskAnalyzer(),
        router=Router(registry=registry),
        executor=executor,
        escalation=EscalationStrategy(executor=executor, registry=registry),
        aggregator=Aggregator(),
        enable_escalation=False,
    )
    result = await pipeline.run("What is 2+2?")
    assert not result.escalated


@pytest.mark.asyncio
async def test_pipeline_with_subject_hint(pipeline):
    result = await pipeline.run(
        "Which element has atomic number 6?",
        subject_hint="college_chemistry",
    )
    assert isinstance(result, OrchestratorResult)
