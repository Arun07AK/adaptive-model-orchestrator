# Model Orchestrator

## What This Is
An intelligent multi-model orchestration system that routes tasks to specialized open-source LLMs. Portfolio project proving smart routing beats single-model inference at the same cost.

## Architecture
5-layer async pipeline: Analyzer → Decomposer → Router → Executor+Escalation → Aggregator

## Commands
- `pytest` — run all tests
- `python scripts/setup_models.py` — download MLX models
- `python scripts/run_benchmarks.py` — run benchmark suite
- `python scripts/train_router.py` — fine-tune routing classifier

## Key Files
- `src/types.py` — all shared types (Domain, ModelConfig, ExecutionResult, etc.)
- `src/models/registry.py` — model configs and lookup
- `src/orchestrator/pipeline.py` — end-to-end orchestration
- `src/benchmarks/harness_adapter.py` — lm-evaluation-harness integration

## Rules
- All inference calls go through `InferenceBackend` protocol (never call MLX/LiteLLM directly from pipeline code)
- Tests use mock backends from `tests/conftest.py` — never require actual model downloads to pass
- Never add Co-Authored-By lines in git commits
