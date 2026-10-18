# Laptop AI Assistant: GIGABYTE AORUS MASTER 16 AM6H

A local RAG hardware assistant for the GIGABYTE AORUS MASTER 16 AM6H, with content moderation at
every pipeline stage, targeting a **4GB VRAM** consumer laptop.

- **No RAG frameworks.** Scraping, parsing, chunking, the vector index, BM25, retrieval, prompt
  building and moderation are plain Python, with numpy for the vector math.
- **Inference uses llama.cpp** (`llama-cpp-python`) for both generation and embeddings.
- The environment is managed with **uv**.
- Supports Traditional Chinese, English and mixed zh/en questions, with streaming answers.

| | Result | Details |
|---|---|---|
| RAG answer accuracy (independent 50-question set) | **0.909**, up from a 0.705 baseline | [benchmark §4](benchmark_moderation.md#4-rag-quality) |
| Harmful requests blocked (blind held-out set) | **0.912** (injection 1.0) | [benchmark §2](benchmark_moderation.md#2-moderation-results) |
| Legitimate questions refused by moderation (held-out) | 5.7% | [benchmark §2.2](benchmark_moderation.md#22-end-to-end-full-pipeline-llm-included) |
| Poisoned knowledge base: poison reaching the LLM | 0% with moderation, 75% without | [benchmark §2.3](benchmark_moderation.md#23-poisoned-knowledge-base-adversarial-4-of-the-task) |
| Moderation overhead | +79 ms TTFT p50, 0 MB extra VRAM | [benchmark §3](benchmark_moderation.md#3-performance-and-resources) |
| **VRAM measured on NVIDIA T4 (NVML, process peak)** | **2378 MiB** (2006 MiB with `n_ubatch=128`), < 4096 ✅ | [§ 4GB VRAM](#4gb-vram-feasibility) |

## Quick start

```bash
uv sync                                   # Python 3.12 env + deps (builds llama-cpp-python)
uv run python scripts/download_models.py  # GGUF models -> models/ (~1.9 GB)
uv run python scripts/prepare_data.py     # data/specs.json (key-value) -> chunks.jsonl
uv run python scripts/build_index.py      # embed chunks -> data/processed/index.npz
uv run aorus-chat --explain               # streaming chat; --explain prints moderation decisions + timings
```

- The parsed key-value spec data is committed as `data/specs.json`, so setup works offline.
- `prepare_data.py --fetch` re-scrapes both product pages, re-parses them and rewrites `data/specs.json`. The scraped HTML goes to `data/raw/`, which is not committed. A re-scrape on 2026-10-18 reproduced the committed file exactly.
- gigabyte.com is behind Akamai. It needs browser headers and an HTTP/2 client: `requests` gets 403, so the scraper uses httpx with HTTP/2.
- On NVIDIA, build with CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" uv sync`. See `notebooks/colab_4gb_verify.ipynb`.

Example session (actual output):
```
> 這台的顯示卡是什麼？
根據提供的規格資料，這台筆記型電腦的顯示晶片（GPU）依版本不同而有所差異：
*   **BZH 版**：NVIDIA GeForce RTX 5090 Laptop GPU
*   **BYH 版**：NVIDIA GeForce RTX 5080 Laptop GPU
*   **BXH 版**：NVIDIA GeForce RTX 5070 Ti Laptop GPU

> 跟 ROG Strix SCAR 16 比哪台好？
抱歉，我無法提供其他品牌產品的規格、價格或比較，只能介紹 AORUS MASTER 16 AM6H。您可以詢問 AORUS MASTER 16 AM6H 的處理器、顯示卡、螢幕、記憶體、連接埠、電池等規格。
```

## Architecture

```
query ─► normalise + embed once (Qwen3-Embedding, CPU) ─┬─► [1] Input guard ── refuse/redirect (no LLM call)
                                                        │      rules: injection, competitor entities+intent,
                                                        │      sensitive keywords+stance; kNN over intent
                                                        │      prototypes; off-topic fallback
                                                        ▼
        hybrid retrieval: dense (numpy cosine) + BM25 (zh char uni/bigrams + en words)
        scope pre-filter: only chunks whose product == AM6H are ever scored
        + sub-queries for multi-part questions
                                                        ▼
        [2] Retrieval filter: drop chunks with competitor mentions / instruction-like text / sensitive content
        no-data threshold ── "not in the product data" (no LLM call)
        context selection: hybrid cutoff ∪ dense top-k ∪ sub-query best, max 4
                                                        ▼
        [3] Generation guard: grounded system prompt, <context> declared as data, labelled key-value facts
        Qwen3.5-2B Q4_K_M on GPU, streaming
                                                        ▼
        [4] Output guard: release clause by clause; competitor/sensitive → fallback;
            ungrounded number → clause removed + notice
                                                        ▼
        [5] Audit: logs/audit.jsonl (per-stage category, action, reason, matched rules, latency)
```

### Data and chunking
- [parser.py](src/aorus_assistant/rag/parser.py) reads the spec comparison table: **3 SKUs × 17 fields**. BZH = RTX 5090, BYH = RTX 5080, BXH = RTX 5070 Ti; only the GPU differs between them. The zh-TW and English pages are merged by row, so every field carries both names, e.g. `顯示晶片 / Video Graphics`.
- [chunker.py](src/aorus_assistant/rag/chunker.py) splits by field, not by length, into 21 chunks:
  - 16 chunks for fields that are identical across SKUs.
  - 3 per-SKU GPU chunks.
  - A SKU-lineup chunk.
  - A performance-overview chunk (CPU + GPU + RAM + storage) for workload questions.
  - Chunks also carry colloquial aliases, used for BM25 only.
- [attributes.py](src/aorus_assistant/rag/attributes.py) turns packed spec lines into labelled key-value facts for the prompt. For example, "OLED WQXGA (2560×1600) 240Hz, 1ms…" becomes `更新率 Refresh rate: 240Hz` and `解析度 Resolution: 2560×1600`. This stopped the model from answering "1600 Hz".

### Moderation
- Policy, categories, edge cases and the third-party-brand rules are in [moderation_policy.md](moderation_policy.md).
- The maintainable lists live in `data/policy/*.yaml`: brands/aliases, scope vocabulary, sensitive topics, injection patterns and intent prototypes. Adding a competitor alias needs no code change.

## Model choice

| Role | Model | Quant | File | Device |
|---|---|---|---|---|
| Generation | `unsloth/Qwen3.5-2B-GGUF` | Q4_K_M | 1281 MB | GPU, all layers |
| Embedding | `Qwen/Qwen3-Embedding-0.6B-GGUF` | Q8_0 | 639 MB | CPU (keeps VRAM for the LLM) |

**Generation model: 2B instead of 0.8B.** Tested with `eval_rag.py --llm-file <gguf>` on 30 answerable and 9 unanswerable questions, using the same prompt and context:

| LLM (Q4_K_M) | Answer acc. | No-data refusal | Decode (Apple M4) |
|---|---|---|---|
| Qwen3.5-0.8B | 0.867 | 0.556 | ~70–90 tok/s |
| **Qwen3.5-2B** | **1.000** | **1.000** | ~45–55 tok/s |

- 0.8B hallucinated answers to unanswerable questions, e.g. a price or a heat-pipe count. Q8_0 made the same errors, so the limit is model capacity, not quantisation.
- **Q4_K_M** is the quantisation sweet spot for 2B here: 1.28 GB of weights leaves room for the KV cache and compute buffers inside 4 GB.
- Qwen3.5 is a hybrid architecture: only 6 of its 24 layers are full attention, the rest are linear-attention layers with a fixed-size state. The KV cache is therefore only 48 MiB at 4096 context.

**Embedding model: Qwen3-Embedding-0.6B.** Compared with bge-m3, retrieval quality was the same: hybrid Hit@1 0.967 and MRR 0.983 for both. Qwen3-Embedding wins because:
- Its file is half the size: 639 MB vs 1158 MB.
- It is instruction-aware.
- The same query embedding is reused by the moderation kNN, so moderation needs no extra model call.

## 4GB VRAM feasibility

GPU allocation is static after load. It is the same for every query until `n_ctx` is exhausted, so the peak equals the loaded size. llama.cpp accounting from [scripts/measure_memory.py](scripts/measure_memory.py):

| GPU buffer | MiB |
|---|---|
| Weights | 1211 |
| Compute buffer (`n_ubatch` 512 → 128) | 493 → 123 |
| KV cache (n_ctx 4096) | 48 |
| Recurrent state | 19 |
| **Total** | **1771 → 1402** |

- On an NVIDIA GPU, add the CUDA context, which llama.cpp does not count.
- The embedding model and the token-embedding table run on the CPU (~2.0 GB of RAM).
- Moderation adds no GPU memory.
**Verification on NVIDIA.** [notebooks/colab_4gb_verify.ipynb](notebooks/colab_4gb_verify.ipynb) runs the same scripts on a Colab T4 and reads the real VRAM via NVML, sampled every 20 ms during full answers. Raw output: `logs/memory_colab_*.json`.

| Colab T4, NVML | `n_ubatch` 512 | `n_ubatch` 128 |
|---|---|---|
| **This process, peak while answering** | **2378 MiB** | **2006 MiB** |
| Whole device, peak (includes 447 MiB held by another process before our models loaded) | 2828 MiB | 2456 MiB |
| of which: GPU compute buffer that llama.cpp reserved for the CPU embedder (op offload) | 456 MiB | 456 MiB |
| Decode / TTFT (335–885-token prompts) | 73–81 tok/s / 109–425 ms | 72–81 tok/s / 129–515 ms |

- **Both configurations pass**: peak < 4096 MiB, even counting the whole device.
- A T4 cannot be hard-capped at 4 GB, so the check is the measured peak.
- The measurement exposed something the Mac could not: with a CUDA build, llama.cpp offloads batch ops of the CPU-resident embedder and reserves 456 MiB of VRAM for them. `EmbedConfig.op_offload=False` now disables this. The expected process peak is then ~1.9 GB (~1.55 GB with `n_ubatch=128`). *Re-measurement with the fix: pending.*

## Performance

Measured on an Apple M4 laptop GPU. Absolute numbers are indicative; see [benchmark §3](benchmark_moderation.md#3-performance-and-resources).

| Metric | Moderation off | Moderation on |
|---|---|---|
| TTFT p50 (to first text the user sees) | 1130 ms | 1209 ms |
| Decode | 44.8 tok/s | 44.7 tok/s |
| Moderation compute per request (p50) | — | 27 ms (input guard 0.2 ms) |

## Evaluation

```bash
uv run pytest                                                       # 63 unit tests, no LLM needed
uv run python eval_moderation.py [--input-only] [--cases-dir tests/cases_holdout]
uv run python eval_rag.py [--qa | --retrieval-only | --llm-file X.gguf]
uv run python benchmark.py --runs 3
uv run python scripts/measure_memory.py [--n-ubatch 128]
```

- Test sets are in `tests/cases/` (main + `rag_qa.jsonl`) and `tests/cases_holdout/`. Schema and file list: [tests/cases/README.md](tests/cases/README.md).
- The moderation and RAG test sets were written by a separate agent from the policy document alone, with no access to the code, lexicons or prototypes.
- The held-out set was written after tuning, so it gives the generalisation estimate.
- Full results, version history, adversarial evaluation and every failure case: **[benchmark_moderation.md](benchmark_moderation.md)**.

## Project layout

```
src/aorus_assistant/
  config.py       every tunable, with the calibration that set it
  pipeline.py     end-to-end orchestration (moderation on/off), streaming via callback
  cli.py          `aorus-chat`
  rag/            scraper, parser, attributes (key-value), chunker, tokenize, index (numpy + BM25),
                  embedder, retriever (hybrid, scope mask, sub-queries, context selection)
  llm/            model loading (llama.cpp), prompt + context formatting, streaming
  moderation/     normalize, lexicon, input_guard, semantic (kNN), retrieval_filter,
                  output_guard, responses, audit
  bench/          TTFT/TPS timing, VRAM (NVML) / RAM sampling, llama.cpp buffer parsing
data/specs.json   parsed key-value specs: 3 SKUs × 17 fields, zh + en field names (source of truth)
data/policy/      brands, scope vocabulary, sensitive topics, injection patterns, intent prototypes
tests/            unit tests + JSONL case sets
eval_moderation.py  eval_rag.py  benchmark.py  scripts/  notebooks/
```

## Known limitations

- Moderation recall is bounded by vocabulary and prototypes. Held-out misses include an unknown alias, regional bias, leetspeak in sensitive words and a résumé request; see [benchmark §6](benchmark_moderation.md#6-failure-cases-v31-nothing-omitted). A small classifier on the existing embedding is the next step.
- The output guard removes clauses with numbers that are correct but not in the data (e.g. "3.5mm"). It cannot catch a grounded number attached to the wrong attribute.
- A 2B model still makes reasoning errors over correct context, such as mixing up left and right ports.

## Troubleshooting

- `ModuleNotFoundError: aorus_assistant` on macOS: some setups (e.g. an iCloud-synced Desktop) set the `hidden` flag on files inside `.venv`, and Python skips hidden `.pth` files.
  - Fix it with `chflags -R nohidden .venv`, or prefix commands with `PYTHONPATH=src`.
  - `pytest` is unaffected, because `pythonpath = ["src"]` is set in pyproject.
- `VIRTUAL_ENV=... does not match` warning: an outer conda/venv is active. It is harmless; `conda deactivate` removes it.
