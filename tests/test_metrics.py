import json
import pytest
from src.benchmarks.metrics import MetricsCollector


def test_record_single_question():
    collector = MetricsCollector()
    collector.record(benchmark="mmlu", question_id="q1", correct=True, model_used="deepseek-r1-7b", latency_ms=150.0, escalated=False)
    assert collector.total_questions("mmlu") == 1
    assert collector.accuracy("mmlu") == 1.0


def test_accuracy_calculation():
    collector = MetricsCollector()
    collector.record("mmlu", "q1", correct=True, model_used="m1", latency_ms=100, escalated=False)
    collector.record("mmlu", "q2", correct=False, model_used="m1", latency_ms=100, escalated=False)
    collector.record("mmlu", "q3", correct=True, model_used="m2", latency_ms=100, escalated=False)
    collector.record("mmlu", "q4", correct=True, model_used="m1", latency_ms=100, escalated=False)
    assert collector.accuracy("mmlu") == 0.75


def test_escalation_rate():
    collector = MetricsCollector()
    collector.record("mmlu", "q1", True, "m1", 100, escalated=False)
    collector.record("mmlu", "q2", True, "m2", 200, escalated=True)
    collector.record("mmlu", "q3", False, "m1", 100, escalated=False)
    collector.record("mmlu", "q4", True, "m2", 300, escalated=True)
    assert collector.escalation_rate("mmlu") == 0.5


def test_avg_latency():
    collector = MetricsCollector()
    collector.record("gsm8k", "q1", True, "m1", 100, False)
    collector.record("gsm8k", "q2", True, "m1", 200, False)
    collector.record("gsm8k", "q3", True, "m1", 300, False)
    assert collector.avg_latency_ms("gsm8k") == 200.0


def test_model_usage_distribution():
    collector = MetricsCollector()
    collector.record("mmlu", "q1", True, "math-7b", 100, False)
    collector.record("mmlu", "q2", True, "math-7b", 100, False)
    collector.record("mmlu", "q3", True, "code-16b", 100, False)
    collector.record("mmlu", "q4", True, "general-7b", 100, False)
    dist = collector.model_distribution("mmlu")
    assert dist["math-7b"] == 2
    assert dist["code-16b"] == 1
    assert dist["general-7b"] == 1


def test_to_summary():
    collector = MetricsCollector()
    collector.record("mmlu", "q1", True, "m1", 100, False)
    collector.record("mmlu", "q2", False, "m1", 200, True)
    summary = collector.to_summary("mmlu")
    assert summary["benchmark"] == "mmlu"
    assert summary["total_questions"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["escalation_rate"] == 0.5
    assert summary["avg_latency_ms"] == 150.0


def test_to_json_serializable():
    collector = MetricsCollector()
    collector.record("mmlu", "q1", True, "m1", 100, False)
    summary = collector.to_summary("mmlu")
    json_str = json.dumps(summary)
    assert "mmlu" in json_str
