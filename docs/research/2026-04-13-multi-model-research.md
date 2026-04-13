# Multi-Model Orchestration Research — April 13, 2026

## The Full Vision

Route each task to the best specialist open-source model, THEN have a stronger model review/correct critical decisions. The combined output beats any single model.

## Architecture Pattern: Route + Review

```
Task → Classifier → Specialist (generates) → Confidence Check → Reviewer (corrects if needed) → Output
```

Proven by research:
- **Mixture of Agents** (Together.ai, ICML 2025): 6 open-source models beat GPT-4o (65.1% vs 57.5% on AlpacaEval)
- **LLM Cascades / FrugalGPT** (Stanford 2023): GPT-4-level quality at 90% cost reduction
- **AutoMix** (2024): Small model's self-confidence triggers escalation
- **LLM Debate** (MIT 2023): Multi-model debate → 10-20% improvement on math/reasoning

## Models Available on Groq Free Tier (verified April 13, 2026)

| Model ID | Size | Best For |
|----------|------|----------|
| `qwen/qwen3-32b` | 32B | Math reasoning (CoT, `/no_think` support) |
| `llama-3.3-70b-versatile` | 70B | General knowledge, reviewing |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 17B | Code, multimodal |
| `llama-3.1-8b-instant` | 8B | Fast routing/classification |

Rate limits: ~6000 TPM per model on free tier.

## Qwen3 `/no_think` Mode

- Append `/no_think` to user message to disable chain-of-thought
- Use for MCQ (want direct letter answer)
- Keep thinking enabled for math reasoning (GSM8K)
- Groq does NOT expose `enable_thinking` param — use the suffix

## Reviewer Best Practices (from literature)

1. Reviewer receives: original question + specialist's full reasoning + final answer
2. Break evaluation into specific criteria (factually correct? logically sound? complete?)
3. Output a structured diff (what changed and why), not a full rewrite
4. Only override when specific flaw found — "don't fix what isn't broken"
5. Ask for reasoning before the verdict — improves judge accuracy

## Tool Stack Decision

For portfolio project, **custom Python** is better than frameworks:
- More impressive in interviews ("I built it" vs "I used LangGraph")
- Existing codebase already has the pipeline, executor, routing
- Just need to add the review layer

Production-grade alternatives (for reference):
- LangGraph for orchestration graphs
- DSPy 3.0 for prompt auto-optimization
- RouteLLM for trained routing
- LM-Polygraph for uncertainty estimation

## Current Benchmark Results (before review layer)

| Benchmark | Single (7B) | Orchestrated (routing only) |
|-----------|-------------|---------------------------|
| MMLU | 60.0% | 66.0% |
| GSM8K | 26.7% | 70.0% |
| ARC | 93.3% | 93.3% |

## Expected After Review Layer

| Benchmark | Orchestrated | + Review | Why |
|-----------|-------------|----------|-----|
| MMLU | 66% | 70-75% | Reviewer catches wrong MCQ answers |
| GSM8K | 70% | 80-85% | Reviewer catches math errors in reasoning |
| ARC | 93% | 95%+ | Reviewer corrects the 2 wrong answers |
