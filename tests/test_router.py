import pytest
from src.orchestrator.router import Router
from src.models.registry import ModelRegistry
from src.types import Domain, CostTier, TaskAnalysis, Complexity


@pytest.fixture
def router():
    return Router(registry=ModelRegistry())


def test_route_math_to_math_model(router):
    analysis = TaskAnalysis(text="Solve x^2 = 4", domain=Domain.MATH, complexity=Complexity.SIMPLE, confidence=0.9)
    decision = router.route(analysis)
    assert decision.model.domain == Domain.MATH


def test_route_code_to_code_model(router):
    analysis = TaskAnalysis(text="Write a binary search function", domain=Domain.CODE, complexity=Complexity.SIMPLE, confidence=0.9)
    decision = router.route(analysis)
    assert decision.model.domain == Domain.CODE


def test_route_general_to_general_model(router):
    analysis = TaskAnalysis(text="What is the capital of France?", domain=Domain.GENERAL, complexity=Complexity.SIMPLE, confidence=0.85)
    decision = router.route(analysis)
    assert decision.model.domain == Domain.GENERAL


def test_route_prefers_cheapest_model(router):
    analysis = TaskAnalysis(text="What is 2+2?", domain=Domain.GENERAL, complexity=Complexity.SIMPLE, confidence=0.95)
    decision = router.route(analysis)
    assert decision.model.cost_tier == CostTier.LOCAL


def test_route_low_confidence_uses_stronger_model(router):
    analysis = TaskAnalysis(text="ambiguous question", domain=Domain.MATH, complexity=Complexity.SIMPLE, confidence=0.3)
    decision = router.route(analysis, prefer_stronger=True)
    assert decision.model.size_b > 7


def test_route_reason_is_populated(router):
    analysis = TaskAnalysis(text="What is 2+2?", domain=Domain.MATH, complexity=Complexity.SIMPLE, confidence=0.95)
    decision = router.route(analysis)
    assert len(decision.reason) > 0
