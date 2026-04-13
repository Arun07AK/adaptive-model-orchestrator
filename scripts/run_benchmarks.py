"""CLI runner for benchmark evaluation of the Adaptive Model Orchestrator.

Usage:
    python scripts/run_benchmarks.py --config single --quick
    python scripts/run_benchmarks.py --config orchestrated
    python scripts/run_benchmarks.py --config escalation --quick
    python scripts/run_benchmarks.py --config all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.runner import BENCHMARK_SUITE, QUICK_SUITE, BenchmarkConfig, save_results
from src.models.registry import ModelRegistry
from src.orchestrator.aggregator import Aggregator
from src.orchestrator.analyzer import TaskAnalyzer
from src.orchestrator.escalation import EscalationStrategy
from src.orchestrator.executor import Executor
from src.orchestrator.pipeline import OrchestratorPipeline
from src.orchestrator.router import Router
from src.types import Domain, RoutingDecision, TaskAnalysis


# ---------------------------------------------------------------------------
# Pipeline factory helpers
# ---------------------------------------------------------------------------

def _make_executor() -> Executor:
    """Build executor using available backends (graceful fallback if no keys)."""
    backends: dict = {}

    try:
        from src.models.local import MLXBackend
        backends["mlx"] = MLXBackend()
    except Exception:
        pass

    try:
        from src.models.api import LiteLLMBackend
        backends["groq"] = LiteLLMBackend()
        backends["together"] = LiteLLMBackend()
    except Exception:
        pass

    if not backends:
        raise RuntimeError(
            "No inference backends available. "
            "Install mlx-lm or set GROQ_API_KEY / TOGETHER_API_KEY."
        )

    return Executor(backends=backends)


class SingleModelRouter(Router):
    """Router that always selects qwen2.5-7b (baseline single-model config)."""

    def route(self, analysis: TaskAnalysis, prefer_stronger: bool = False) -> RoutingDecision:
        model = self._registry.get_by_name("qwen2.5-7b")
        if model is None:
            model = self._registry.get_cheapest(Domain.GENERAL)
        return RoutingDecision(model=model, reason="single-model baseline: qwen2.5-7b")


def build_pipeline(config_name: str) -> OrchestratorPipeline:
    registry = ModelRegistry()
    executor = _make_executor()

    if config_name == "single":
        return OrchestratorPipeline(
            analyzer=TaskAnalyzer(),
            router=SingleModelRouter(registry=registry),
            executor=executor,
            escalation=EscalationStrategy(executor=executor, registry=registry),
            aggregator=Aggregator(),
            enable_escalation=False,
            enable_decomposition=False,
        )

    if config_name == "orchestrated":
        return OrchestratorPipeline(
            analyzer=TaskAnalyzer(),
            router=Router(registry=registry),
            executor=executor,
            escalation=EscalationStrategy(executor=executor, registry=registry),
            aggregator=Aggregator(),
            enable_escalation=False,
            enable_decomposition=False,
        )

    if config_name == "escalation":
        return OrchestratorPipeline(
            analyzer=TaskAnalyzer(),
            router=Router(registry=registry),
            executor=executor,
            escalation=EscalationStrategy(executor=executor, registry=registry, threshold=0.6),
            aggregator=Aggregator(),
            enable_escalation=True,
            enable_decomposition=False,
        )

    raise ValueError(f"Unknown config: {config_name!r}. Choose single/orchestrated/escalation.")


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def run_config(
    config_name: str,
    suite: list[BenchmarkConfig],
    output_dir: str = "data/results",
) -> dict:
    try:
        import lm_eval  # noqa: F401
        from lm_eval import simple_evaluate
    except ImportError:
        print(
            "[ERROR] lm-evaluation-harness not installed.\n"
            "  pip install lm-eval"
        )
        sys.exit(1)

    from src.benchmarks.harness_adapter import OrchestratorLM

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Benchmarks: {[b.name for b in suite]}")
    print(f"{'='*60}")

    pipeline = build_pipeline(config_name)
    lm = OrchestratorLM(pipeline=pipeline)

    all_results: dict = {"config": config_name, "benchmarks": {}}

    for bench in suite:
        print(f"\n  Running {bench.name} (fewshot={bench.num_fewshot}, limit={bench.limit})...")
        try:
            results = simple_evaluate(
                model=lm,
                tasks=[bench.task_name],
                num_fewshot=bench.num_fewshot,
                limit=bench.limit,
                log_samples=False,
            )
            all_results["benchmarks"][bench.name] = results.get("results", {})
            # Print a brief summary
            bench_results = results.get("results", {}).get(bench.task_name, {})
            acc = bench_results.get("acc,none", bench_results.get("exact_match,none", "N/A"))
            print(f"    {bench.name}: accuracy={acc}")
        except Exception as exc:
            print(f"    [FAILED] {bench.name}: {exc}")
            all_results["benchmarks"][bench.name] = {"error": str(exc)}

    # Save
    filepath = save_results(all_results, config_name, output_dir)
    print(f"\n  Results saved to: {filepath}")
    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark suite against the Adaptive Model Orchestrator"
    )
    parser.add_argument(
        "--config",
        choices=["single", "orchestrated", "escalation", "all"],
        default="orchestrated",
        help="Pipeline configuration to evaluate (default: orchestrated)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced QUICK_SUITE (100/50/50 samples) instead of full suite",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory to write JSON result files (default: data/results)",
    )
    args = parser.parse_args()

    suite = QUICK_SUITE if args.quick else BENCHMARK_SUITE

    configs_to_run = (
        ["single", "orchestrated", "escalation"]
        if args.config == "all"
        else [args.config]
    )

    summary: dict = {}
    for cfg in configs_to_run:
        results = run_config(cfg, suite, output_dir=args.output_dir)
        summary[cfg] = results

    # Print comparison table when running all
    if args.config == "all" and len(configs_to_run) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
