from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Domain(Enum):
    MATH = "math"
    CODE = "code"
    REASONING = "reasoning"
    GENERAL = "general"


class CostTier(Enum):
    """Ordered by cost: LOCAL (0) < FREE_API (1) < PAID_API (2)."""
    LOCAL = 0
    FREE_API = 1
    PAID_API = 2


class Complexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str           # "mlx", "groq", "together"
    domain: Domain
    size_b: float
    cost_tier: CostTier
    model_id: str           # Provider-specific model identifier
    ram_gb: float = 0.0


@dataclass
class TaskAnalysis:
    text: str
    domain: Domain
    complexity: Complexity
    confidence: float
    subtasks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingDecision:
    model: ModelConfig
    reason: str


@dataclass
class ExecutionResult:
    text: str
    confidence: float
    model_used: str
    latency_ms: float
    token_count: int
    log_probs: list[float] | None = None


@dataclass
class OrchestratorResult:
    text: str
    model_used: str
    escalated: bool
    escalation_model: str | None = None
    total_latency_ms: float = 0.0
    confidence: float = 0.0
