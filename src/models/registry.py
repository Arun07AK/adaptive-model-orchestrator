from __future__ import annotations

from src.types import CostTier, Domain, ModelConfig


_DEFAULT_MODELS: list[ModelConfig] = [
    # --- Math specialist (Groq free tier, genuinely stronger than 7B) ---
    ModelConfig(
        name="qwen3-32b",
        provider="groq",
        domain=Domain.MATH,
        size_b=32.0,
        cost_tier=CostTier.FREE_API,
        model_id="qwen/qwen3-32b",
    ),
    # --- Code specialist (Groq free tier) ---
    ModelConfig(
        name="llama-4-scout-17b",
        provider="groq",
        domain=Domain.CODE,
        size_b=17.0,
        cost_tier=CostTier.FREE_API,
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
    ),
    ModelConfig(
        name="qwen2.5-7b",
        provider="mlx",
        domain=Domain.GENERAL,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        ram_gb=5.0,
    ),
    ModelConfig(
        name="qwen2.5-7b-reasoning",
        provider="mlx",
        domain=Domain.REASONING,
        size_b=7.0,
        cost_tier=CostTier.LOCAL,
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        ram_gb=5.0,
    ),
    ModelConfig(
        name="llama-3.3-70b",
        provider="groq",
        domain=Domain.GENERAL,
        size_b=70.0,
        cost_tier=CostTier.FREE_API,
        model_id="llama-3.3-70b-versatile",
    ),
    ModelConfig(
        name="llama-3.1-8b",
        provider="groq",
        domain=Domain.GENERAL,
        size_b=8.0,
        cost_tier=CostTier.FREE_API,
        model_id="llama-3.1-8b-instant",
    ),
]

_ESCALATION_MODEL_NAME = "llama-3.3-70b"


class ModelRegistry:
    def __init__(self, models: list[ModelConfig] | None = None) -> None:
        self._models = models if models is not None else list(_DEFAULT_MODELS)
        self._by_name: dict[str, ModelConfig] = {m.name: m for m in self._models}

    def all_models(self) -> list[ModelConfig]:
        return list(self._models)

    def get_by_name(self, name: str) -> ModelConfig | None:
        return self._by_name.get(name)

    def get_models_for_domain(self, domain: Domain) -> list[ModelConfig]:
        return [m for m in self._models if m.domain == domain]

    def get_cheapest(self, domain: Domain) -> ModelConfig:
        domain_models = self.get_models_for_domain(domain)
        if not domain_models:
            raise ValueError(f"No models registered for domain {domain}")
        return min(domain_models, key=lambda m: (m.cost_tier.value, m.size_b))

    def get_strongest(self, domain: Domain) -> ModelConfig:
        domain_models = self.get_models_for_domain(domain)
        if not domain_models:
            raise ValueError(f"No models registered for domain {domain}")
        return max(domain_models, key=lambda m: m.size_b)

    def get_escalation_model(self) -> ModelConfig:
        model = self._by_name.get(_ESCALATION_MODEL_NAME)
        if model is None:
            raise ValueError(f"Escalation model '{_ESCALATION_MODEL_NAME}' not in registry")
        return model
