import pytest
from src.orchestrator.decomposer import Decomposer
from src.types import Complexity, Domain, TaskAnalysis
from tests.conftest import MockBackend


@pytest.fixture
def backend():
    mock = MockBackend()
    mock.set_response(
        "qwen2.5-7b", "Decompose",
        "1. Parse the input CSV file\n2. Calculate statistics\n3. Generate chart",
        0.9,
    )
    return mock


@pytest.fixture
def decomposer(backend):
    return Decomposer(backend=backend, decomposer_model_name="qwen2.5-7b")


def test_simple_task_not_decomposed():
    decomposer = Decomposer(backend=MockBackend(), decomposer_model_name="qwen2.5-7b")
    analysis = TaskAnalysis(text="What is 2+2?", domain=Domain.MATH, complexity=Complexity.SIMPLE, confidence=0.95)
    assert decomposer.should_decompose(analysis) is False


def test_complex_task_should_decompose():
    decomposer = Decomposer(backend=MockBackend(), decomposer_model_name="qwen2.5-7b")
    analysis = TaskAnalysis(text="Build a REST API with authentication, database, and tests", domain=Domain.CODE, complexity=Complexity.COMPLEX, confidence=0.8)
    assert decomposer.should_decompose(analysis) is True


@pytest.mark.asyncio
async def test_decompose_returns_subtasks(decomposer):
    analysis = TaskAnalysis(text="Parse CSV, compute stats, make chart", domain=Domain.CODE, complexity=Complexity.COMPLEX, confidence=0.8)
    subtasks = await decomposer.decompose(analysis)
    assert len(subtasks) >= 2
    assert all(isinstance(s, str) for s in subtasks)
