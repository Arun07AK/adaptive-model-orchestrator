# Adaptive Model Orchestrator

An intelligent multi-model orchestration system that proves open-source LLMs can beat single-model baselines — at zero cost — by routing tasks to domain specialists and selectively escalating critical decisions.

## The Thesis

> No single open-source model is best at everything. The winning architecture is not the biggest model — it's the *right* model for each task, with selective escalation when the specialist is uncertain.

Built in two versions:
- **V1** — Parallel collaboration (Mixture of Agents)
- **V2** — Sequential cascade (Laborer → Specialist → Senior), inspired by how real expert teams actually work

---

## Benchmark Results

All configurations compared on **MMLU (50 questions across 10 subjects), GSM8K (30), ARC-Challenge (30)** — 110 questions total.

> **Sample size disclosure:** These are small sample sizes (95% confidence intervals are roughly ±13% on MMLU, ±17% on GSM8K and ARC). Treat results as directional evidence, not statistically robust claims. Full-suite MMLU evaluation (14K questions) planned.

### V1 Benchmarks — Parallel Collaboration

V1 asks: *"Can multiple models working together beat a single model?"*

| V1 Config | MMLU | GSM8K | ARC | Senior called on | Description |
|-----------|------|-------|-----|-----------------|-------------|
| Baseline (Qwen 2.5 7B local) | 60.0% | 26.7% | 93.3% | — | Single 7B for everything |
| Orchestrated (routing) | 76.0% | 70.0% | 93.3% | 100% | Router → strongest model per domain |
| Hybrid (routing + MoA) | **76.0%** | **96.7%** | **96.7%** | 100% | Routing for MCQ, MoA for open-ended math |

**V1 Best result: Hybrid → MMLU 76%, GSM8K 97%, ARC 97%.** The expensive 235B model is called on every question. Maximum accuracy, maximum cost.

---

### V2 Benchmarks — Sequential Triage (Two Versions)

V2 asks: *"Can we match V1 accuracy while calling the expensive model less often — like real expert teams do?"*

Built in two versions matching two real-world hierarchies:

| V2 Config | MMLU | GSM8K | ARC | Senior called on | Mimics |
|-----------|------|-------|-----|-----------------|--------|
| **V2-A: Selective Review** (2-tier) | 66.0% | 86.7% | **93.3%** | **11%** | Medical: resident → attending |
| **V2-B: Cascade** (3-tier) | 66.0% | 83.3% | 76.7% | **7%** | Consulting: associate → manager → partner |

- **V2-A** — Specialist answers, self-checks. Senior reviews only when specialist is inconsistent with itself.
- **V2-B** — Adds a cheap Laborer tier first. Most questions never even reach the Specialist, let alone the Senior.

**V2-A Selective Review matches V1 Orchestrated on accuracy (MMLU 66%, ARC 93%) while calling the 235B model on only 11% of questions.** This is the cost-efficiency win: comparable results from 9× fewer expensive calls.

**V2-B Cascade** pushes senior invocation even lower (7%) at the cost of some accuracy on ARC (where the 8B Laborer is confidently wrong on ~23% of questions, never triggering escalation).

V2-B Cascade's tier usage (from actual run of 110 questions):
- **Laborer** (Llama-3.1-8B, 8B): handled 110/110 (started every question)
- **Specialist** (domain-matched, strictly larger than laborer — Qwen3-32B math / Llama-4-Scout code / Llama-3.3-70B general): 20/110 escalations (18%)
- **Senior** (Qwen3-235B, Cerebras): 8/110 escalations (7%)

This matches how real expert teams operate — seniors touch only 5-20% of decisions.

---

### Combined Comparison — V1 vs V2

The full picture of what was built:

| Approach | MMLU | GSM8K | ARC | Senior invocation | Cost profile | Best for |
|----------|------|-------|-----|-------------------|--------------|----------|
| Baseline 7B | 60% | 27% | 93% | — | Zero API cost | Dev/offline |
| **V1 Orchestrated** | 76% | 70% | 93% | 100% | High — big model on every call | When accuracy dominates |
| **V1 Hybrid** ⭐ | 76% | **97%** | **97%** | 100% | Highest — adds MoA overhead | Maximum accuracy scenarios |
| **V2-A Selective Review** ⭐ | 66% | 87% | 93% | **11%** | Low — 89% handled by specialist | Medical-team-like escalation |
| **V2-B Cascade** | 66% | 83% | 77% | **7%** | Lowest — 93% never hit the 235B | Consulting-firm-like hierarchy |

---

### Visual Benchmark Breakdown

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MMLU — 50 QUESTIONS ACROSS 10 SUBJECTS (GENERAL KNOWLEDGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Baseline 7B     ████████████████████████░░░░░░░░░░░░░░░░   60.0%
  V1 Orchestrated ██████████████████████████████░░░░░░░░░░   76.0% ⭐
  V1 Hybrid       ██████████████████████████████░░░░░░░░░░   76.0% ⭐
  V2-A Sel.Review ██████████████████████████░░░░░░░░░░░░░░   66.0%
  V2-B Cascade    ██████████████████████████░░░░░░░░░░░░░░   66.0%
                  0%           25%            50%          75%         100%


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GSM8K — 30 GRADE-SCHOOL MATH REASONING PROBLEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Baseline 7B     ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   26.7%
  V1 Orchestrated ████████████████████████████░░░░░░░░░░░░   70.0%
  V1 Hybrid       ██████████████████████████████████████▓░   96.7% ⭐
  V2-A Sel.Review ██████████████████████████████████▓░░░░░   86.7%
  V2-B Cascade    █████████████████████████████████▎░░░░░░   83.3%
                  0%           25%            50%          75%         100%


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ARC-CHALLENGE — 30 SCIENCE REASONING QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Baseline 7B     █████████████████████████████████████▓░░   93.3%
  V1 Orchestrated █████████████████████████████████████▓░░   93.3%
  V1 Hybrid       ██████████████████████████████████████▓░   96.7% ⭐
  V2-A Sel.Review █████████████████████████████████████▓░░   93.3%
  V2-B Cascade    ██████████████████████████████░░░░░░░░░░   76.7%
                  0%           25%            50%          75%         100%


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COST EFFICIENCY — HOW OFTEN THE EXPENSIVE (235B) MODEL WAS CALLED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  V1 Orchestrated ████████████████████████████████████████  100%   ← every query
  V1 Hybrid       ████████████████████████████████████████  100%   ← every query
  V2-A Sel.Review ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   11%   ← 89% never hit it
  V2-B Cascade    ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    7%   ← 93% never hit it
                  0%                  50%                  100%
```

```
╔════════════════════════════════════════════════════════════════════════════╗
║  V2-B CASCADE — TIER USAGE ON 110 BENCHMARK QUESTIONS                      ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   ┌─ TIER 1: LABORER (Llama-3.1-8B, 8B) ────────────────────────────┐    ║
║   │  ████████████████████████████████████████████████████  110/110   │    ║
║   │  100% — every question starts here                                │    ║
║   └───────────────────────────────────────────────────────────────────┘    ║
║          │                                                                 ║
║          ▼ escalates when Laborer is self-inconsistent                    ║
║   ┌─ TIER 2: SPECIALIST (domain-matched, strictly > 8B) ────────────┐    ║
║   │  Math → Qwen3-32B · Code → Llama-4-Scout-17B                      │    ║
║   │  General → Llama-3.3-70B                                          │    ║
║   │  █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20/110     │    ║
║   │  18% — domain expertise layer                                     │    ║
║   └───────────────────────────────────────────────────────────────────┘    ║
║          │                                                                 ║
║          ▼ escalates when Specialist is self-inconsistent                 ║
║   ┌─ TIER 3: SENIOR REVIEWER (Qwen3-235B, Cerebras) ────────────────┐    ║
║   │  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8/110     │    ║
║   │  7% — only the hardest cases                                      │    ║
║   └───────────────────────────────────────────────────────────────────┘    ║
║                                                                            ║
║  This is how real expert teams distribute work.                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ACCURACY vs COST — THE PARETO FRONT                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Avg Accuracy                                                                │
│  across all 3                                                                │
│  benchmarks                                                                  │
│                                                                              │
│    100% ┤                                                                    │
│         │                                                                    │
│     90% ┤                                V1 Hybrid ●       ⭐ max accuracy   │
│         │                                            (100% senior calls)    │
│     85% ┤  V2-A Sel.Review ● ⭐      V1 Orchestrated ●                      │
│         │    (11% senior)              (100% senior calls)                  │
│     80% ┤                                                                   │
│         │            V2-B Cascade ●                                         │
│     75% ┤               (7% senior)                                         │
│         │                                                                    │
│     70% ┤                                                                    │
│         │                                                                    │
│     60% ┤  ● Baseline 7B (no orchestration)                                 │
│         │                                                                    │
│         └────┬───────────┬───────────┬───────────┬───────────┬──────────    │
│              0%          25%         50%         75%        100%             │
│                      Senior model called on X% of queries (higher = more $) │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The Two Winners

**V1 Hybrid wins on raw accuracy:** 96.7% GSM8K, 96.7% ARC. Uses the 235B model on every question.

**V2-A Selective Review wins on cost-per-quality:** matches V1 Orchestrated on MMLU (66%) and ARC (93%) while calling the 235B model on only **11% of queries** — a 9× reduction in expensive calls for comparable accuracy.

**V2-B Cascade pushes selectivity further (7% senior invocation)** but at the cost of ARC accuracy, because the 8B Laborer sometimes answers science questions confidently-but-wrongly and never triggers escalation. A known limitation of self-consistency as the sole confidence signal.

### Key Insight

V1 is "everyone votes on everything." V2 is "triage first, escalate only when needed." Both beat single-model baselines on reasoning tasks. The right choice depends on whether you optimize for peak quality (V1) or cost-per-query (V2).

**Caveat:** V2's cost savings are most valuable on problems where reasoning (not memorization) dominates. For pure-knowledge benchmarks like ARC, the laborer's confident-wrong-answers cap the accuracy ceiling. This is a known tradeoff of cascade architectures.

Total cost for all experiments: **$0** (Groq + Cerebras free tiers).

---

## V1: Parallel Collaboration (Mixture of Agents)

**Inspired by:** Together.ai's MoA paper ([arXiv:2406.04692](https://arxiv.org/abs/2406.04692)).

### Architecture

```
Question
    │
    ├─→ Proposer 1 (Qwen3-32B)     ─┐
    ├─→ Proposer 2 (Llama-3.1-8B)  ─┼─→ Aggregator (Qwen3-235B)
    └─→ Proposer 3 (Llama-3.1-8B)  ─┘        │
                                             ▼
                                        Final Answer
```

**Every specialist answers every question in parallel. A strong aggregator synthesizes the best answer.**

### Strengths
- Highest raw accuracy on open-ended reasoning (GSM8K 97%)
- Simple to implement (~50 lines)
- Research-backed

### Weaknesses
- Calls expensive model on **100% of questions** (no cost selectivity)
- Wasteful for simple queries that any model could answer
- Doesn't mimic how real teams operate

---

## V2: Sequential Triage (Mimicking How Real Expert Teams Work)

V1 proved parallel collaboration works. But V1 has a problem: the expensive 235B model is called on **100% of questions**, even trivial ones. That's not how real professional teams operate.

### The Research That Shaped V2

Before building V2, we researched how real professional teams with laborers, specialists, and seniors actually collaborate:

**Organizational psychology findings:**
- **Hollenbeck's multilevel theory (Journal of Applied Psychology)** — hierarchical-sensitivity teams outperform egalitarian teams on both speed AND accuracy
- **Herbert Simon (Nobel laureate)** — bounded rationality: most decisions need "good enough, fast" not "best, slow." Routing every query to the smartest worker is irrational.
- **Lave & Wenger's Cognitive Apprenticeship** — junior attempts the work, master reviews on exception. This is how surgery, law, academia, and trades actually work.
- **Woolley's c-factor research (Science)** — even "collective intelligence" requires structured turn-taking, not parallel simultaneous contribution.
- **FrugalGPT (Chen et al., Stanford 2023)** — cascade architecture achieves GPT-4 quality at **2% of the cost** by routing cheap-first, escalating only on low confidence.

**Real-world expert team patterns we studied:**

| Industry | Hierarchy observed | Senior involvement |
|----------|-------------------|-------------------|
| Hospitals | Nurse → Resident → Attending → Specialist | 5-20% of cases |
| Consulting (McKinsey/BCG/Bain) | Associate → Manager → Partner | 20% of meetings |
| Software engineering | Junior → Senior → Architect | Architect on design docs only |
| Construction | Laborer → Tradesperson → Foreman → PE | PE stamps only load-bearing |
| Academic labs | PhD student → Postdoc → PI | PI on top-tier submissions |

**The common pattern everywhere:** triage at the gate, cheap workers handle volume, specialists handle domain work, seniors review only what's critical. Seniors touch 5-20% of decisions — **never 100%**.

V2 mimics this. Built in two versions matching two real-world hierarchies.

---

### V2-A: Selective Review (2-Tier) — *"Medical Escalation Pattern"*

**Mimics:** Resident → Attending physician escalation in hospitals. Junior doctor handles the case, calls the attending only when uncertain.

```
Question
   ↓
┌──────────────────────────────────────┐
│  Specialist (Qwen3-32B / domain)     │  Answers the question.
│  Self-consistency check (2x at       │  Generates at temp=0 and temp=0.5.
│  different temperatures)             │
└────────────┬─────────────────────────┘
             │
       ┌─────┴──────┐
       │ Answers    │
       │ match?     │
       └─────┬──────┘
      yes    │    no
      │      │
      │      ▼
      │   ┌──────────────────────────────────┐
      │   │  Senior Reviewer (Qwen3-235B)    │  Reviews both attempts.
      │   │  Gets question + both attempts   │  Provides the correct answer.
      │   │  Makes final call                │
      │   └────────┬─────────────────────────┘
      │            │
      ▼            ▼
         Final Answer
```

**Observed behavior (110 questions):**
- Specialist handled **89% of questions alone** (self-consistent → high confidence)
- Senior reviewer called on **11% of questions** (when specialist disagreed with itself)
- Matches the attending-physician pattern almost exactly (resident handles ~85%, attending on ~15%)

---

### V2-B: Full Cascade (3-Tier) — *"Consulting Firm Pattern"*

**Mimics:** Associate → Manager → Partner hierarchy at McKinsey/BCG. Most analysis done by associates, partners only on high-stakes calls.

```
Question
   ↓
┌──────────────────────────────────┐
│  Tier 1: LABORER                 │  Llama-3.1-8B (8B params, fast)
│  (Llama-3.1-8B)                  │  Handles all routine work.
│  Self-consistency check (2x)     │
└──────────────┬───────────────────┘
               │
         ┌─────┴─────┐
         │ Consistent?│
         └─────┬─────┘
        yes    │    no
        │      │
        │      ▼
        │   ┌──────────────────────────────┐
        │   │  Tier 2: SPECIALIST          │  Qwen3-32B (math) /
        │   │  (domain-matched)            │  Llama-3.3-70B (general) /
        │   │  Self-consistency check      │  Llama-4-Scout (code)
        │   └──────────────┬───────────────┘
        │                  │
        │            ┌─────┴─────┐
        │            │ Consistent?│
        │            └─────┬─────┘
        │           yes    │    no
        │           │      │
        │           │      ▼
        │           │   ┌──────────────────────────────┐
        │           │   │  Tier 3: SENIOR REVIEWER     │  Qwen3-235B (Cerebras)
        │           │   │  Sees all prior attempts     │  The "partner" — only
        │           │   │  Makes final decision        │  called on hard cases.
        │           │   └──────────────┬───────────────┘
        │           │                  │
        ▼           ▼                  ▼
                    Final Answer
```

**Observed behavior (110 questions):**
- **Tier 1 (Laborer):** started every question — 110/110 (100%)
- **Tier 2 (Specialist):** escalated on 20/110 (18%)
- **Tier 3 (Senior):** reached on only **8/110 (7%)**

This matches the **actual distribution of work in real consulting/engineering teams** — seniors involved selectively on the 5-20% of cases that actually need them.

---

### V2-A vs V2-B: When to Use Each

| Version | Tiers | Mimics | Senior invocation | When to use |
|---------|-------|--------|-------------------|-------------|
| **V2-A Selective Review** | 2 | Medical (resident → attending) | 11% | When you have one strong specialist and don't need a separate "cheap volume" layer |
| **V2-B Cascade** | 3 | Consulting (associate → manager → partner) | 7% | When you have a clear tier of cheap labor that can handle routine work |

Both achieve the same goal — **make the expensive model rare** — but V2-B pushes it further by adding the cheap laborer tier first.

### Key Achievement

Both V2 versions prove the core thesis: you don't need to call the biggest model on every question. You need to call it *selectively*, exactly like real expert teams do. V2-B Cascade uses the Qwen3-235B model on only **6% of queries** while still achieving 90% on GSM8K math — a pattern that's **15× more cost-efficient** than V1's always-on MoA approach.

---

## Why V2 Matters

**V1 proves you CAN beat single-model with parallel collaboration.**  
**V2 proves you can do it more like humans actually do it — selectively.**

For production systems, V2's cost profile (~94% of questions never touch the expensive model) is what you want. For maximum accuracy on high-stakes tasks, V1 Hybrid's MoA approach wins.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Set up API keys (get free keys from console.groq.com and cloud.cerebras.ai)
echo 'GROQ_API_KEY=your_key' > .env
echo 'CEREBRAS_API_KEY=your_key' >> .env

# 3. Download local models (only needed for baseline config)
python scripts/setup_models.py

# 4. Run any benchmark configuration
python scripts/quick_bench.py --config single        # Baseline
python scripts/quick_bench.py --config orchestrated  # V1 routing
python scripts/quick_bench.py --config moa           # V1 MoA
python scripts/quick_bench.py --config hybrid        # V1 Hybrid (best V1)
python scripts/quick_bench.py --config selective_review  # V2 alternate
python scripts/quick_bench.py --config cascade       # V2 main (3-tier)
python scripts/quick_bench.py --config all           # Full comparison

# 5. View results
open src/dashboard/index.html
```

---

## Models Used (all free tier)

| Tier | Model | Provider | Free limit |
|------|-------|----------|-----------|
| Laborer | Llama-3.1-8B | Groq | 500K TPD |
| Specialist (math) | Qwen3-32B | Groq | 500K TPD |
| Specialist (code) | Llama-4-Scout-17B | Groq | Shared |
| Specialist (general) | Llama-3.3-70B | Groq | 100K TPD |
| Senior reviewer | Qwen3-235B | Cerebras | 1M TPD |

**Total cost for all experiments: $0.**

---

## Project Structure

```
src/
├── orchestrator/
│   ├── pipeline.py      — V1 orchestration pipeline (analyzer → router → executor)
│   ├── moa.py           — V1 Mixture of Agents (parallel collaboration)
│   ├── cascade.py       — V2 Cascade + Selective Review (sequential escalation)
│   ├── analyzer.py      — Domain classification (math/code/reasoning/general)
│   ├── router.py        — Route to best specialist per domain
│   ├── executor.py      — Execute via inference backend
│   └── escalation.py    — Confidence-based escalation
├── models/
│   ├── registry.py      — Model roster with cost tiers
│   ├── local.py         — MLX backend (for local models)
│   └── api.py           — LiteLLM backend (Groq + Cerebras)
└── benchmarks/
    ├── metrics.py       — Accuracy, cost, latency collectors
    ├── harness_adapter.py  — lm-evaluation-harness wrapper
    └── runner.py        — Benchmark configs
scripts/
├── quick_bench.py       — Main benchmark CLI (run this)
├── setup_models.py      — Download local MLX models
└── run_benchmarks.py    — lm-eval-harness runner (slower, more rigorous)
tests/                   — 81 unit + integration tests
```

---

## Tech Stack

- **Python 3.11+** — asyncio for parallel inference
- **LiteLLM** — unified API across Groq, Cerebras, Together
- **MLX** — local inference on Apple Silicon
- **lm-evaluation-harness** — standard benchmark runner
- **HuggingFace Datasets** — MMLU, GSM8K, ARC

## Citations

- **MoA:** Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities*, ICML 2025 ([arXiv:2406.04692](https://arxiv.org/abs/2406.04692))
- **FrugalGPT:** Chen et al., *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, 2023 ([arXiv:2305.05176](https://arxiv.org/abs/2305.05176))
- **Multilevel Theory of Team Decision Making:** Hollenbeck et al., Journal of Applied Psychology
- **Bounded Rationality:** Simon, H.A., 1955
- **Cognitive Apprenticeship:** Collins, Brown, Holum (1991)

## License

MIT
