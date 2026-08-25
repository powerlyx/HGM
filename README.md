# HGM

<p align="center">
  <img src="https://img.shields.io/static/v1?label=Paper&message=coming%20soon&color=blue" height="23">
  <img src="https://img.shields.io/static/v1?label=Benchmarks&message=LoCoMo%20%7C%20LongMemEval&color=green" height="23">
  <img src="https://img.shields.io/static/v1?label=Python&message=3.10%2B&color=orange" height="23">
</p>

This repository contains the official implementation of
**HGM: Long-Term Memory Management for LLM Agents with Hierarchical Graph-Structured Memory**.

HGM manages long-term memory for LLM agents over long-horizon dialogues. Conversation
histories are organized into hierarchical graph-structured memory across
**Turn → Episode → Topic** granularities; retrieval fuses semantic, lexical, temporal
and recency signals and routes budgets under uncertainty. The repository ships
end-to-end evaluation pipelines on two long-term memory benchmarks,
[LoCoMo](https://github.com/snap-research/locomo) and
[LongMemEval](https://github.com/xiaowu0162/longmemeval), together with ablation
tooling for reproducing the paper experiments.

## ✨ Highlights

- **Hierarchical memory** — raw turns are kept as Turn memories, summarized per session
  into Episodes, and clustered across sessions into Topics, with mutually reinforcing
  cross-layer retrieval.
- **Hybrid retrieval with uncertainty routing** — vector semantics, BM25 lexical
  matching, temporal alignment and recency are fused; score distribution and
  multi-query consistency drive low/mid/high tiers that adapt candidate budgets,
  final evidence caps and fusion weights.
- **Evidence packet** — retrieved items are deduplicated, ranked and compressed into a
  traceable evidence packet; insufficient evidence triggers escalation and verification
  retrieval.
- **Temporal reasoning** — query plans carry explicit temporal intent; date-related
  questions trigger time-aware retrieval and evidence sufficiency checks.
- **Fixed evaluation protocols** — LoCoMo: structured-evidence answers + CORRECT/WRONG
  judge; LongMemEval: 7-step CoT answers + majority-vote judge, so results are directly
  comparable.
- **Reproducibility** — seeds, query plans, per-question evidence, judge verdicts, token
  usage and timings are all recorded; results can be reused across `--ratio` settings.
- **Ablation tooling** — 8 component combinations (hierarchy / uncertainty routing /
  evidence loop) plus an optional `answer_driven` post-answer self-verification link.

## 🧠 Overview

```text
Conversations (LoCoMo / LongMemEval)
    │
    ├── Turn memory (raw turns + batched keyword/context analysis)
    ├── Episode memory (per-session segmentation & summarization)
    └── Topic memory (cross-episode clustering & summarization)
             │
Question ──> Query Plan ──> uncertainty / temporal routing ──> hierarchical hybrid retrieval
                                                     │
                                      evidence sufficiency / expansion / verification
                                                     │
                                            Evidence packet
                                                     │
                                     Answer LLM ──> lexical metrics / LLM judge
```

## 📁 Project Structure

```text
HGM/
├── main.py                      # Unified entry point (auto-detects LoCoMo / LongMemEval)
├── config.py                    # Global configuration dataclass
├── requirements.txt             # Pinned, verified dependencies
├── data/                        # LoCoMo data (included) and data notes
├── dataset/                     # Parsers and data models for both benchmarks
│   ├── locomo.py
│   └── longmemeval.py
├── memory/                      # Hierarchical memory core
│   ├── memory_system.py         # Building, clustering, hybrid retrieval, uncertainty
│   │                            # routing, state export/restore
│   ├── retriever.py             # OpenAI / SentenceTransformer embedding retriever
│   ├── memory_note.py           # Memory node
│   ├── anchors.py               # Temporal/entity/event anchors (off by default)
│   ├── config.py                # Memory-system configuration & validation
│   └── prompts.py               # Memory-building prompts and parsers
├── eval/                        # Evaluation pipeline
│   ├── agent.py                 # Query plans, evidence selection, sufficiency loop,
│   │                            # answer generation
│   ├── runner.py                # LoCoMo sample loop, caching, metric aggregation
│   ├── longmemeval_runner.py    # LongMemEval per-entry build/cache, result reuse,
│   │                            # concurrency
│   ├── llm_judge.py             # The two fixed judge protocols
│   ├── longmemeval_prompts.py   # LongMemEval answer-check prompts
│   └── metrics.py               # EM/F1/ROUGE/BLEU/METEOR lexical metrics
└── llm/
    └── controller.py            # Multi-backend (openai/ollama/sglang/vllm) controllers
                                 # with token-usage tracking
```

## 🔧 Setup

Requires Python ≥ 3.10 (verified on 3.10.20):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the verified pinned combination. `sentence-transformers`,
`transformers` and `torch` have tightly coupled versions — upgrade them together.

Configure API credentials (any OpenAI-compatible service works):

```bash
export OPENAI_API_KEY="<your-api-key>"
# Optional: point to a custom OpenAI-compatible endpoint; defaults to OpenAI when unset
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

`OPENAI_BASE_URL` applies to the answer/memory LLM and the embedding service; the judge
endpoint can be overridden separately with `--judge-base-url`. Ollama, SGLang and vLLM
are also supported as answer or judge backends (`--llm-backend` / `--judge-backend`).

## 📜 Data

- **LoCoMo**: `data/locomo10.json` is included in this repository (see `data/README.md`
  for provenance and licensing).
- **LongMemEval**: cannot be redistributed due to its official terms; please obtain
  `longmemeval_s_cleaned.json` yourself following `data/README.md` and place it under
  `data/`.

## 🚀 Quick Start

```bash
# 0. Inspect the CLI (no external calls are made)
python main.py --help

# 1. LoCoMo evaluation (fixed protocol: structured-evidence answers + CORRECT/WRONG judge)
python main.py \
  --llm-backend openai --llm-model gpt-4o-mini \
  --dataset data/locomo10.json \
  --output outputs/locomo_main.json

# 2. Low-cost smoke test on a small subset
python main.py \
  --ratio 0.1 --seed 0 \
  --dataset data/locomo10.json \
  --output outputs/locomo_smoke.json

# 3. LongMemEval evaluation (memory is built per entry; build the cache with a small
#    --ratio first)
python main.py \
  --dataset-type longmemeval --dataset data/longmemeval_s_cleaned.json \
  --llm-backend openai --llm-model gpt-4.1-mini-2025-04-14 \
  --judge-backend openai --judge-model gpt-4.1-mini-2025-04-14 \
  --ratio 0.02 --seed 0 \
  --output outputs/lme_smoke.json

# 4. Ablation runs
python main.py --ablation-mode no_uncertainty --seed 0 \
  --dataset data/locomo10.json --output outputs/no_uncertainty_seed0.json
```

## ✅ Self-Check

```bash
python main.py --help
python -m compileall -q main.py config.py memory eval dataset llm
```

The release does not ship a test suite; the commands above verify the installation and
CLI integrity without incurring any API cost.

## ⚠️ Limitations

- Full evaluation requires external models, network access, credentials and the
  associated cost.
- OpenAI-compatible endpoints for answer/memory/embedding are configured via
  `OPENAI_BASE_URL` (official endpoint when unset); the judge endpoint via
  `--judge-base-url`.
- Memory building, retrieval and evaluation live in a few large modules; consider
  splitting them by responsibility for further development.
- LLM outputs, remote services and model versions affect reproducibility; fix the seed,
  model, backend, dataset and query plans when comparing experiments.
- No automatic cache/output cleanup command is provided; verify target directories
  before manual cleanup.
- Embedding-token cost is not counted by the built-in token tracker; reconcile it from
  your service billing.

## ✏️ Citation

If you find this work useful, please cite:

```bibtex
@article{hgm2026,
  title   = {HGM: Long-Term Memory Management for LLM Agents with Hierarchical Graph-Structured Memory},
  author  = {TODO},
  year    = {2026},
  note    = {to be updated upon publication}
}
```

## 🙏 Acknowledgements

- **LoCoMo**: public long-dialogue benchmark from Snap Research, see
  [snap-research/locomo](https://github.com/snap-research/locomo).
- **LongMemEval**: obtained from the official channel of
  [xiaowu0162/longmemeval](https://github.com/xiaowu0162/longmemeval) under its usage
  terms; not redistributed in this repository.
