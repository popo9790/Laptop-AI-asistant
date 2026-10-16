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
TODO

## Results
TODO: see `benchmark_moderation.md`.
