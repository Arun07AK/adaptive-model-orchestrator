from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    name: str
    task_name: str
    num_fewshot: int = 0
    limit: int | None = None


BENCHMARK_SUITE: list[BenchmarkConfig] = [
    BenchmarkConfig(name="mmlu", task_name="mmlu", num_fewshot=5),
    BenchmarkConfig(name="gsm8k", task_name="gsm8k", num_fewshot=5),
    BenchmarkConfig(name="math", task_name="minerva_math", num_fewshot=4),
    BenchmarkConfig(name="humaneval", task_name="humaneval", num_fewshot=0),
    BenchmarkConfig(name="arc_challenge", task_name="arc_challenge", num_fewshot=25),
]

QUICK_SUITE: list[BenchmarkConfig] = [
    BenchmarkConfig(name="mmlu", task_name="mmlu", num_fewshot=5, limit=100),
    BenchmarkConfig(name="gsm8k", task_name="gsm8k", num_fewshot=5, limit=50),
    BenchmarkConfig(name="arc_challenge", task_name="arc_challenge", num_fewshot=25, limit=50),
]


def save_results(
    results: dict,
    config_name: str,
    output_dir: str = "data/results",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{config_name}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath
