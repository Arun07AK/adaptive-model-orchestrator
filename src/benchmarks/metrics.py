from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class QuestionRecord:
    question_id: str
    correct: bool
    model_used: str
    latency_ms: float
    escalated: bool


@dataclass
class BenchmarkRun:
    benchmark: str
    config_name: str = ""
    records: list[QuestionRecord] = field(default_factory=list)


class MetricsCollector:
    def __init__(self) -> None:
        self._records: dict[str, list[QuestionRecord]] = {}

    def record(
        self,
        benchmark: str,
        question_id: str,
        correct: bool,
        model_used: str,
        latency_ms: float,
        escalated: bool,
    ) -> None:
        if benchmark not in self._records:
            self._records[benchmark] = []
        self._records[benchmark].append(
            QuestionRecord(
                question_id=question_id,
                correct=correct,
                model_used=model_used,
                latency_ms=latency_ms,
                escalated=escalated,
            )
        )

    def total_questions(self, benchmark: str) -> int:
        return len(self._records.get(benchmark, []))

    def accuracy(self, benchmark: str) -> float:
        records = self._records.get(benchmark, [])
        if not records:
            return 0.0
        return sum(1 for r in records if r.correct) / len(records)

    def escalation_rate(self, benchmark: str) -> float:
        records = self._records.get(benchmark, [])
        if not records:
            return 0.0
        return sum(1 for r in records if r.escalated) / len(records)

    def avg_latency_ms(self, benchmark: str) -> float:
        records = self._records.get(benchmark, [])
        if not records:
            return 0.0
        return sum(r.latency_ms for r in records) / len(records)

    def model_distribution(self, benchmark: str) -> dict[str, int]:
        records = self._records.get(benchmark, [])
        return dict(Counter(r.model_used for r in records))

    def to_summary(self, benchmark: str) -> dict:
        return {
            "benchmark": benchmark,
            "total_questions": self.total_questions(benchmark),
            "accuracy": round(self.accuracy(benchmark), 4),
            "escalation_rate": round(self.escalation_rate(benchmark), 4),
            "avg_latency_ms": round(self.avg_latency_ms(benchmark), 2),
            "model_distribution": self.model_distribution(benchmark),
        }
