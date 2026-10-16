# Laptop AI Assistant: GIGABYTE AORUS MASTER 16 AM6H

Local RAG hardware assistant with content moderation, designed for a **4GB VRAM** target.
No RAG frameworks: chunking, retrieval, generation, and moderation are plain Python; inference uses **llama.cpp**.

## Quick start

```bash
uv sync                                   # create env + install deps
uv run python scripts/download_models.py  # fetch GGUF models into models/
uv run python scripts/prepare_data.py     # parse spec pages -> specs.json + chunks.jsonl (--fetch to re-download)
uv run python scripts/build_index.py      # embed chunks -> index
uv run aorus-chat                         # streaming chat
```

## Evaluation

```bash
uv run pytest -m "not llm"                # unit tests (no model needed)
uv run python eval_moderation.py          # moderation metrics
uv run python eval_rag.py                 # RAG quality
uv run python benchmark.py                # TTFT / TPS / VRAM
```

## Project layout

```
src/aorus_assistant/
  rag/          scraper, parser, chunker, tokenize, index (numpy + BM25), retriever
  llm/          model loading (llama.cpp), grounded streaming generator
  moderation/   normalize, lexicon, input_guard, retrieval_filter, output_guard, responses, audit
  bench/        TTFT/TPS timing, VRAM/RAM sampling
  pipeline.py   end-to-end orchestration
data/policy/    brands.yaml, sensitive_topics.yaml (maintainable lists)
tests/cases/    JSONL test sets
```

## Model choice & 4GB VRAM feasibility

| Role | Model | Quant | File | Device |
|---|---|---|---|---|
| Generation | `unsloth/Qwen3.5-2B-GGUF` | Q4_K_M | 1281 MB | GPU (all layers) |
| Embedding | `Qwen/Qwen3-Embedding-0.6B-GGUF` | Q8_0 | 639 MB | CPU |

Embedding model comparison on `tests/cases/retrieval.jsonl` (30 zh/en/mixed queries):

| Embedding | dense Hit@1 | hybrid Hit@1 | hybrid MRR | query p50 (CPU) | file |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B Q8_0 (w/ instruction) | 0.933 | 0.967 | 0.983 | 64 ms | 639 MB |
| Qwen3-Embedding-0.6B Q8_0 (no instruction) | 0.933 | 0.933 | 0.967 | 21 ms | 639 MB |
| bge-m3 F16 | 0.933 | 0.967 | 0.983 | 20 ms | 1158 MB |

Generation model comparison (`uv run python eval_rag.py --llm-file <gguf>`; 30 answerable +
9 unanswerable questions; same prompt and context; latency on Apple Silicon Metal, indicative only):

| LLM | file | answer acc. | no-data refusal | false refusal | TTFT p50 | decode |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B Q4_K_M | 533 MB | 0.867 | 0.556 | 0.0 | ~190-260 ms | ~70-90 tok/s |
| Qwen3.5-0.8B Q8_0 | 812 MB | same errors as Q4_K_M on spot checks | | | | slower |
| **Qwen3.5-2B Q4_K_M** | 1281 MB | **1.0** | **1.0** | 0.0 | ~470-690 ms | ~43-55 tok/s |

0.8B failure modes: answered one SKU when the GPU varies by SKU, misread HDMI/card-reader
questions, and hallucinated values for unanswerable questions (a price, a heat-pipe count).
Q8_0 did not fix them, so the limit is model capacity rather than quantisation.

TODO: VRAM measurements on a 4GB-limited GPU.

## Troubleshooting

- `ModuleNotFoundError: aorus_assistant` on macOS: some setups (e.g. iCloud-synced Desktop) set the
  `hidden` flag on files inside `.venv`, and Python skips hidden `.pth` files. Fix with
  `chflags -R nohidden .venv`. `pytest` is unaffected (`pythonpath = ["src"]` in pyproject).
- `VIRTUAL_ENV=... does not match` warning: an outer conda/venv is active; harmless, or run `conda deactivate`.

## Results
TODO: see `benchmark_moderation.md`.
