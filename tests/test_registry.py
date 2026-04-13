from src.types import Domain, CostTier, ModelConfig
from src.models.registry import ModelRegistry


def test_registry_has_all_domains():
    registry = ModelRegistry()
    for domain in Domain:
        models = registry.get_models_for_domain(domain)
        assert len(models) >= 1, f"No models registered for {domain}"


def test_get_cheapest_for_domain():
    registry = ModelRegistry()
    model = registry.get_cheapest(Domain.MATH)
    assert model.domain == Domain.MATH


def test_get_strongest_for_domain():
    registry = ModelRegistry()
    model = registry.get_strongest(Domain.MATH)
    assert model.size_b >= 30


def test_get_escalation_model():
    registry = ModelRegistry()
    model = registry.get_escalation_model()
    assert model.size_b >= 70
    assert model.cost_tier in (CostTier.FREE_API, CostTier.PAID_API)


def test_get_model_by_name():
    registry = ModelRegistry()
    model = registry.get_by_name("qwen3-32b")
    assert model is not None
    assert model.domain == Domain.MATH


def test_get_model_by_name_not_found():
    registry = ModelRegistry()
    model = registry.get_by_name("nonexistent-model")
    assert model is None


def test_all_models_have_model_id():
    registry = ModelRegistry()
    for model in registry.all_models():
        assert model.model_id, f"{model.name} missing model_id"


def test_local_models_have_ram_estimate():
    registry = ModelRegistry()
    for model in registry.all_models():
        if model.cost_tier == CostTier.LOCAL:
            assert model.ram_gb > 0, f"Local model {model.name} missing ram_gb"
