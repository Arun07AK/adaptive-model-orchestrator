from src.types import (
    Domain, CostTier, Complexity,
    ModelConfig, TaskAnalysis, RoutingDecision,
    ExecutionResult, OrchestratorResult,
)


def test_domain_enum_values():
    assert Domain.MATH.value == "math"
    assert Domain.CODE.value == "code"
    assert Domain.REASONING.value == "reasoning"
    assert Domain.GENERAL.value == "general"


def test_cost_tier_ordering():
    assert CostTier.LOCAL.value < CostTier.FREE_API.value < CostTier.PAID_API.value


def test_model_config_creation():
    config = ModelConfig(
        name="deepseek-r1-distill-qwen-7b",
        provider="mlx",
        domain=Domain.MATH,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        ram_gb=5.0,
    )
    assert config.name == "deepseek-r1-distill-qwen-7b"
    assert config.domain == Domain.MATH
    assert config.cost_tier == CostTier.LOCAL


def test_task_analysis_defaults():
    analysis = TaskAnalysis(
        text="What is 2+2?",
        domain=Domain.MATH,
        complexity=Complexity.SIMPLE,
        confidence=0.95,
    )
    assert analysis.subtasks == []
    assert analysis.complexity == Complexity.SIMPLE


def test_execution_result_creation():
    result = ExecutionResult(
        text="4",
        confidence=0.98,
        model_used="deepseek-r1-distill-qwen-7b",
        latency_ms=150.0,
        token_count=1,
    )
    assert result.log_probs is None
    assert result.confidence == 0.98


def test_orchestrator_result_no_escalation():
    result = OrchestratorResult(
        text="4",
        model_used="deepseek-r1-distill-qwen-7b",
        escalated=False,
        total_latency_ms=200.0,
        confidence=0.98,
    )
    assert result.escalation_model is None
    assert not result.escalated


def test_orchestrator_result_with_escalation():
    result = OrchestratorResult(
        text="The integral evaluates to pi/4",
        model_used="llama-3.3-70b",
        escalated=True,
        escalation_model="llama-3.3-70b",
        total_latency_ms=500.0,
        confidence=0.92,
    )
    assert result.escalated
    assert result.escalation_model == "llama-3.3-70b"
