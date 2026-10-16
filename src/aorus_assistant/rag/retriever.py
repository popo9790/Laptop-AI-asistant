"""Hybrid retriever (dense + BM25) with product-scope filtering applied at retrieval time.

score = w * cosine + (1 - w) * bm25 / max(bm25)
  - cosine keeps its absolute meaning, so `min_score` on the dense part is used as the
    "no relevant data" signal (BM25 max-normalisation is relative to the query and
    would always give the best chunk 1.0).
  - BM25 rescues exact tokens dense models blur: model numbers, "5070 Ti", "HDMI 2.1".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aorus_assistant.config import RetrievalConfig
from aorus_assistant.rag.index import BM25, VectorIndex
from aorus_assistant.rag.tokenize import tokenize
from aorus_assistant.schemas import Chunk, RetrievedChunk


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        index: VectorIndex,
        embed_query: Callable[[list[str]], np.ndarray],
        cfg: RetrievalConfig,
        chunk_filter: Callable[[Chunk], bool] | None = None,
    ):
        if [c.id for c in chunks] != index.ids:
            raise ValueError("chunk order does not match the vector index; rebuild the index")
        self.chunks = chunks
        self.index = index
        self.embed_query = embed_query
        self.cfg = cfg
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        # Scope mask is computed once: filtered chunks are never scored or returned.
        self.allowed = np.array(
            [chunk_filter(c) if chunk_filter else True for c in chunks], dtype=bool
        )

    def score(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        dense = self.index.scores(self.embed_query([query])[0])
        sparse = self.bm25.scores(tokenize(query))
        return dense, sparse

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.cfg.top_k
        dense, sparse = self.score(query)
        sparse_norm = sparse / sparse.max() if sparse.max() > 0 else sparse
        w = self.cfg.dense_weight
        hybrid = np.where(self.allowed, w * dense + (1 - w) * sparse_norm, -np.inf)

        results = []
        for i in np.argsort(-hybrid)[:top_k]:
            if not np.isfinite(hybrid[i]):
                break
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[i],
                    score=float(hybrid[i]),
                    dense_score=float(dense[i]),
                    sparse_score=float(sparse_norm[i]),
                )
            )
        return results

    def has_relevant(self, results: list[RetrievedChunk]) -> bool:
        return bool(results) and results[0].dense_score >= self.cfg.min_score
