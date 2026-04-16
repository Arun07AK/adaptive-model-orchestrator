# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Multi-model LLM orchestration system exploring cost-efficient inference. Routes tasks to domain specialists and selectively escalates to a senior reviewer. Evaluates 7 configs (V1 routing/MoA, V2-A self-consistency, V2-B cascade, V3 cross-model) against two baselines (7B local, 235B ceiling) on MMLU/GSM8K/ARC benchmarks using free-tier APIs (Groq + Cerebras).

## Commands

```bash
pip install -e ".[dev]"                            # Install with dev deps
pytest                                             # Run all 81+ tests (asyncio_mode=auto)
pytest tests/test_cascade.py -v                    # Single test file
ruff check src/ tests/                             # Lint
ruff format src/ tests/                            # Format
mypy src/ tests/                                   # Type-check (strict mode)

# Benchmarks (need .env with GROQ_API_KEY + CEREBRAS_API_KEY)
python scripts/quick_bench.py --config single              # 7B baseline
python scripts/quick_bench.py --config qwen235b_standalone # 235B ceiling
python scripts/quick_bench.py --config orchestrated        # V1 routing
python scripts/quick_bench.py --config hybrid              # V1 MoA hybrid
python scripts/quick_bench.py --config selective_review    # V2-A
python scripts/quick_bench.py --config cascade             # V2-B 3-tier
python scripts/quick_bench.py --config v3_cross_model      # V3 cross-model

python scripts/setup_models.py --check             # Check local MLX models
bash scripts/render_charts.sh                      # Re-render LinkedIn charts via Chrome
```

## Architecture

### Core Pipeline (V1 base: `src/orchestrator/pipeline.py`)

`TaskAnalyzer` (regex domain classification) → `Router` (cheapest or strongest per domain) → `Executor` (dispatches to `InferenceBackend` by provider key) → `EscalationStrategy` (confidence-threshold gating) → `Aggregator` (combines subtask results).

All inference routed through `InferenceBackend` protocol (`src/models/base.py`). Two implementations: `MLXBackend` (local Apple Silicon, model caching with LRU eviction) and `LiteLLMBackend` (Groq/Cerebras/Together via unified API with retry + exponential backoff for rate limits and 503s).

### V2/V3 Variants (`src/orchestrator/cascade.py`)

- **`SelectiveReviewPipeline` (V2-A)** — specialist answers 2× at different temperatures (self-consistency), escalates to senior on disagreement. Tracks `review_count/total_count`.
- **`CascadePipeline` (V2-B)** — 3-tier: Laborer → Specialist → Senior, each with self-consistency check. Tracks per-tier counters.
- **`CrossModelPipeline` (V3)** — two architecturally-orthogonal models (Llama family + Qwen family) answer at temp=0, escalate on disagreement. Stronger confidence signal than self-consistency.

### MoA (`src/orchestrator/moa.py`)

`MixtureOfAgents` — proposers answer sequentially (avoids rate limits), aggregator synthesizes using Together.ai's system prompt. Strips `<think>` tags from proposals before aggregation.

### Model Registry (`src/models/registry.py`)

7 models across 3 providers. `get_cheapest(domain)` sorts by `(cost_tier.value, size_b)`. `get_strongest(domain)` sorts by `size_b`. Escalation model hardcoded to `llama-3.3-70b`.

Cost tier ordering: `LOCAL (0) < FREE_API (1) < PAID_API (2)`.

### Benchmark Runner (`scripts/quick_bench.py`)

Each config has a builder function returning a pipeline-compatible object (all implement `async run(prompt, subject_hint, max_tokens) -> OrchestratorResult`). Answer extraction uses regex: strips `<think>` CoT blocks, then pattern-matches for letter (MCQ) or number (GSM8K: `####`, `\boxed{}`, "answer is" patterns).

Specialist selector (`_select_specialist_fn`) picks models strictly larger than the laborer's 8B size, falling back to laborer itself if none available (cascade gracefully skips that tier).

## Environment

`.env` at project root (gitignored):
```
GROQ_API_KEY=gsk_...        # Free: 500K TPD per model
CEREBRAS_API_KEY=csk_...    # Free: 1M TPD
```

Provider prefix mapping in `LiteLLMBackend`: `groq/`, `together_ai/`, `cerebras/`.

## Key Conventions

- All inference through `InferenceBackend` protocol — never call MLX/LiteLLM directly from pipeline code
- Tests use `MockBackend` from `tests/conftest.py` with `set_response(model_name, prompt_contains, response, confidence)` — never require actual model downloads
- `_normalize_answer()` in `cascade.py` strips `<think>` tags, extracts letter or number — used for self/cross-model consistency comparison
- Never add Co-Authored-By lines in git commits
- API retry: 12 attempts, 20-180s backoff for queue/high-traffic errors, 3-36s for rate limits
- GitHub Pages deployed at `arun07ak.github.io/adaptive-model-orchestrator` (auto-redirects to animation)
