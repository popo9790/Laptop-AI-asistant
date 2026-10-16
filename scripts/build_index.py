"""Embed data/processed/chunks.jsonl -> data/processed/index.npz.

    uv run python scripts/build_index.py
"""

import argparse
import time

from aorus_assistant.config import Config
from aorus_assistant.rag.embedder import Embedder
from aorus_assistant.rag.index import VectorIndex
from aorus_assistant.rag.retriever import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    cfg = Config()
    chunks = load_chunks(cfg.chunks_path)
    embedder = Embedder(cfg.embed)
    t0 = time.perf_counter()
    vecs = embedder.embed_docs([c.text for c in chunks])
    elapsed = time.perf_counter() - t0
    VectorIndex(vecs, [c.id for c in chunks], cfg.embed.filename).save(cfg.index_path)
    print(
        f"{len(chunks)} chunks, dim={vecs.shape[1]}, {elapsed:.2f}s "
        f"({elapsed / len(chunks) * 1000:.0f} ms/chunk) -> {cfg.index_path}"
    )


if __name__ == "__main__":
    main()
