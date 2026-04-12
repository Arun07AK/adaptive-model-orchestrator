# Adaptive Model Orchestrator

An intelligent multi-model orchestration system that routes tasks to specialized open-source LLMs, proving that smart routing beats single-model inference — at the same cost.

## The Thesis

> Same hardware. Same cost. Zero extra API spend. Just smarter routing — and 10-15% higher accuracy.

Most people pick one LLM and run everything through it. But no single model is best at everything. This system routes each task to the specialist model best suited for it:

- **Math** -> DeepSeek-R1-Distill (94.3% MATH benchmark)
- **Code** -> Qwen 2.5 Coder (88% HumanEval)
- **General** -> Qwen 2.5 (75 MMLU)
- **Hard questions** -> Escalate to Llama 3.3 70B

## Architecture

```
Question -> Analyzer -> Router -> Executor -> [Escalation] -> Result
               |           |         |            |
           Classify     Pick      Run the      If low
           domain      best       model       confidence,
                      model                   try bigger
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Download local models (requires M-series Mac)
python scripts/setup_models.py

# 3. Run quick benchmark (subset)
python scripts/run_benchmarks.py --config all --quick

# 4. View results
open src/dashboard/index.html
```

## Benchmark Results

Three configurations compared on MMLU, GSM8K, MATH, HumanEval, and ARC-Challenge:

| Config | Description | Cost |
|--------|-------------|------|
| Single Model | Everything on Qwen 2.5 7B | $0 |
| Orchestrated | Smart routing to specialists | $0 |
| + Escalation | Routing + selective escalation to 70B | ~$0 |

Results are saved to `data/results/` and visualized in the dashboard.

## Project Structure

- `src/orchestrator/` — Pipeline components (analyzer, router, executor, escalation, aggregator)
- `src/models/` — Model registry and inference backends (MLX local, LiteLLM API)
- `src/benchmarks/` — lm-evaluation-harness adapter, metrics, runner
- `src/dashboard/` — Results visualization
- `scripts/` — CLI tools for benchmarks, model setup, router training
- `tests/` — Unit and integration tests

## Tech Stack

Python, MLX, LiteLLM, lm-evaluation-harness, asyncio
