#!/usr/bin/env python3
"""Fast benchmark runner that directly evaluates the orchestrator pipeline.

Bypasses lm-evaluation-harness overhead. Loads questions from HuggingFace datasets,
runs them through the pipeline, checks answers, reports metrics.

Usage:
    python scripts/quick_bench.py --config single
    python scripts/quick_bench.py --config orchestrated
    python scripts/quick_bench.py --config all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.models.local import MLXBackend
from src.models.api import LiteLLMBackend
from src.models.registry import ModelRegistry
from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.router import Router
from src.orchestrator.executor import Executor
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.aggregator import Aggregator
from src.orchestrator.pipeline import OrchestratorPipeline
from src.orchestrator.moa import MixtureOfAgents
from src.types import Domain, RoutingDecision, TaskAnalysis


# --- Pipeline builders ---

class SingleModelRouter(Router):
    """Baseline: everything through local Qwen 2.5 7B."""
    def route(self, analysis: TaskAnalysis, prefer_stronger: bool = False) -> RoutingDecision:
        model = self._registry.get_by_name("qwen2.5-7b")
        if model is None:
            model = self._registry.get_cheapest(Domain.GENERAL)
        return RoutingDecision(model=model, reason="single-model baseline")


class StrongestRouter(Router):
    """Orchestrated: always route to the strongest model per domain."""
    def route(self, analysis: TaskAnalysis, prefer_stronger: bool = False) -> RoutingDecision:
        model = self._registry.get_strongest(analysis.domain)
        return RoutingDecision(
            model=model,
            reason=f"orchestrated: strongest {analysis.domain.value} → {model.name}",
        )


def build_pipeline(config: str) -> OrchestratorPipeline:
    mlx_backend = MLXBackend()
    api_backend = LiteLLMBackend()
    registry = ModelRegistry()
    executor = Executor(backends={"mlx": mlx_backend, "groq": api_backend, "together": api_backend})
    escalation = EscalationStrategy(executor=executor, registry=registry, threshold=0.6)

    if config == "single":
        router = SingleModelRouter(registry=registry)
        enable_esc = False
    elif config == "orchestrated":
        router = StrongestRouter(registry=registry)
        enable_esc = False
    elif config == "escalation":
        router = StrongestRouter(registry=registry)
        enable_esc = True
    else:
        raise ValueError(f"Unknown config: {config}")

    return OrchestratorPipeline(
        analyzer=TaskAnalyzer(),
        router=router,
        executor=executor,
        escalation=escalation,
        aggregator=Aggregator(),
        enable_escalation=enable_esc,
    )


def build_moa() -> MixtureOfAgents:
    api_backend = LiteLLMBackend()
    registry = ModelRegistry()
    executor = Executor(backends={"groq": api_backend})

    proposer_models = [
        registry.get_by_name("qwen3-32b"),
        registry.get_by_name("llama-3.3-70b"),
        registry.get_by_name("llama-4-scout-17b"),
        registry.get_by_name("llama-3.1-8b"),
    ]
    # Filter out None in case a model isn't in registry
    proposer_models = [m for m in proposer_models if m is not None]

    aggregator = registry.get_by_name("llama-3.3-70b")

    return MixtureOfAgents(
        executor=executor,
        proposer_models=proposer_models,
        aggregator_model=aggregator,
    )


# --- MMLU ---

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "college_chemistry",
    "college_computer_science", "college_physics",
    "high_school_mathematics", "high_school_physics",
    "machine_learning", "computer_security", "philosophy",
]  # 10 diverse subjects covering math, code, reasoning, general

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_mmlu_prompt(item: dict) -> str:
    choices = item["choices"]
    prompt = f"{item['question']}\n"
    for i, choice in enumerate(choices):
        prompt += f"{ANSWER_MAP[i]}. {choice}\n"
    prompt += "\nAnswer with just the letter (A, B, C, or D):"
    return prompt


def extract_answer(text: str) -> str | None:
    # Strip <think>...</think> blocks from CoT models (Qwen3)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    text = text.upper()
    # Try to find A, B, C, or D at the start
    match = re.match(r'^([A-D])', text)
    if match:
        return match.group(1)
    # Check for "answer is X" or "answer: X" patterns
    answer_match = re.search(r'(?:ANSWER|ANSWER IS|ANSWER:)\s*([A-D])', text)
    if answer_match:
        return answer_match.group(1)
    # Try the last single letter A-D in the text
    last_match = re.findall(r'\b([A-D])\b', text)
    if last_match:
        return last_match[-1]
    return None


async def run_mmlu(pipeline: OrchestratorPipeline | MixtureOfAgents, limit_per_subject: int = 5) -> dict:
    correct = 0
    total = 0
    models_used: dict[str, int] = {}
    total_latency = 0.0

    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception:
            try:
                ds = load_dataset("lukaemon/mmlu", subject, split="test")
            except Exception:
                print(f"  Skipping {subject} (dataset not found)")
                continue

        items = list(ds)[:limit_per_subject]

        for item in items:
            prompt = format_mmlu_prompt(item)
            expected = ANSWER_MAP[item["answer"]]

            result = await pipeline.run(prompt, subject_hint=subject, max_tokens=100)

            predicted = extract_answer(result.text)
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            total += 1

            model = result.model_used
            models_used[model] = models_used.get(model, 0) + 1
            total_latency += result.total_latency_ms

        print(f"  {subject}: {correct}/{total} so far")

    accuracy = correct / total if total > 0 else 0
    return {
        "benchmark": "mmlu",
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(total_latency / total, 1) if total > 0 else 0,
        "models_used": models_used,
    }


# --- GSM8K ---

def extract_number(text: str) -> str | None:
    """Extract the final number from a GSM8K answer.

    Priority: #### format > </think> content > \\boxed{} > "answer is" > last number.
    """
    # Strip <think> block to get clean answer
    clean = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()

    # Look for #### number (standard GSM8K format)
    hash_match = re.findall(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', clean.replace(",", ""))
    if hash_match:
        return hash_match[-1]

    # Look for "the answer is X" pattern
    answer_match = re.search(r'(?:the answer is|answer is|answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', clean, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).replace(",", "")

    # Look for boxed answers: \boxed{72}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', clean)
    if boxed:
        nums = re.findall(r'-?\d+(?:\.\d+)?', boxed[-1].replace(",", ""))
        if nums:
            return nums[-1]

    # If there's content after </think>, prefer numbers from there
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', after_think.replace(",", ""))
        if numbers:
            return numbers[-1]

    # Fall back to last number in clean text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', clean.replace(",", ""))
    return numbers[-1] if numbers else None


async def run_gsm8k(pipeline: OrchestratorPipeline | MixtureOfAgents, limit: int = 30) -> dict:
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=False)
    items = list(ds)[:limit]

    correct = 0
    total = 0
    models_used: dict[str, int] = {}
    total_latency = 0.0

    for item in items:
        prompt = (
            "Solve this math problem step by step. "
            "After your reasoning, write the final answer as a single number "
            "in the format: #### [number]\n\n"
            f"{item['question']}"
        )
        expected = extract_number(item["answer"].split("####")[-1].strip())

        result = await pipeline.run(prompt, subject_hint="elementary_mathematics", max_tokens=1024)

        predicted = extract_number(result.text)
        is_correct = predicted is not None and expected is not None and predicted == expected
        if is_correct:
            correct += 1
        total += 1

        model = result.model_used
        models_used[model] = models_used.get(model, 0) + 1
        total_latency += result.total_latency_ms

    accuracy = correct / total if total > 0 else 0
    print(f"  gsm8k: {correct}/{total}")
    return {
        "benchmark": "gsm8k",
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(total_latency / total, 1) if total > 0 else 0,
        "models_used": models_used,
    }


# --- ARC ---

async def run_arc(pipeline: OrchestratorPipeline | MixtureOfAgents, limit: int = 30) -> dict:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=False)
    items = list(ds)[:limit]

    correct = 0
    total = 0
    models_used: dict[str, int] = {}
    total_latency = 0.0

    for item in items:
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        prompt = f"{item['question']}\n"
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
        prompt += "\nAnswer with just the letter:"

        expected = item["answerKey"]

        result = await pipeline.run(prompt, subject_hint="arc_challenge", max_tokens=100)

        predicted = extract_answer(result.text)
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        total += 1

        model = result.model_used
        models_used[model] = models_used.get(model, 0) + 1
        total_latency += result.total_latency_ms

    accuracy = correct / total if total > 0 else 0
    print(f"  arc: {correct}/{total}")
    return {
        "benchmark": "arc_challenge",
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(total_latency / total, 1) if total > 0 else 0,
        "models_used": models_used,
    }


# --- Main ---

async def run_all_benchmarks(config: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    if config == "moa":
        runner = build_moa()
    else:
        runner = build_pipeline(config)

    start = time.time()
    mmlu = await run_mmlu(runner, limit_per_subject=5)
    gsm8k = await run_gsm8k(runner, limit=30)
    arc = await run_arc(runner, limit=30)
    elapsed = time.time() - start

    results = {
        "config": config,
        "total_time_s": round(elapsed, 1),
        "benchmarks": {
            "mmlu": mmlu,
            "gsm8k": gsm8k,
            "arc_challenge": arc,
        },
    }

    # Print summary
    print(f"\n--- {config} Summary ---")
    for name, bench in results["benchmarks"].items():
        print(f"  {name}: {bench['accuracy']*100:.1f}% ({bench['correct']}/{bench['total']}) "
              f"avg={bench['avg_latency_ms']:.0f}ms  models={bench['models_used']}")
    print(f"  Total time: {elapsed:.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark runner")
    parser.add_argument("--config", choices=["single", "orchestrated", "moa", "all"],
                        required=True)
    args = parser.parse_args()

    configs = ["single", "orchestrated", "moa"] if args.config == "all" else [args.config]

    all_results = {}
    for config in configs:
        result = asyncio.run(run_all_benchmarks(config))
        all_results[config] = result

        # Save individual result
        out_dir = Path("data/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        filepath = out_dir / f"{config}_{ts}.json"
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {filepath}")

    # Print comparison if multiple configs
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        header = f"{'Benchmark':<15}"
        for cfg in configs:
            header += f" {cfg:>15}"
        print(header)
        print("-" * len(header))

        for bench in ["mmlu", "gsm8k", "arc_challenge"]:
            row = f"{bench:<15}"
            for cfg in configs:
                acc = all_results[cfg]["benchmarks"][bench]["accuracy"] * 100
                row += f" {acc:>14.1f}%"
            print(row)


if __name__ == "__main__":
    main()
