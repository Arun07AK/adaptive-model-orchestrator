import pytest
from src.orchestrator.aggregator import Aggregator
from src.types import ExecutionResult


def test_aggregate_single_result():
    aggregator = Aggregator()
    result = ExecutionResult(text="42", confidence=0.95, model_used="math-7b", latency_ms=100.0, token_count=1)
    aggregated = aggregator.aggregate([result])
    assert aggregated.text == "42"
    assert aggregated.confidence == 0.95
    assert aggregated.total_latency_ms == 100.0


def test_aggregate_multiple_results():
    aggregator = Aggregator()
    results = [
        ExecutionResult(text="Step 1 done", confidence=0.9, model_used="code-16b", latency_ms=200.0, token_count=5),
        ExecutionResult(text="Step 2 done", confidence=0.85, model_used="math-7b", latency_ms=150.0, token_count=5),
        ExecutionResult(text="Step 3 done", confidence=0.92, model_used="general-7b", latency_ms=100.0, token_count=5),
    ]
    aggregated = aggregator.aggregate(results)
    assert "Step 1 done" in aggregated.text
    assert "Step 2 done" in aggregated.text
    assert "Step 3 done" in aggregated.text
    assert aggregated.confidence == 0.85
    assert aggregated.total_latency_ms == 450.0


def test_aggregate_empty_raises():
    aggregator = Aggregator()
    with pytest.raises(ValueError, match="No results to aggregate"):
        aggregator.aggregate([])
