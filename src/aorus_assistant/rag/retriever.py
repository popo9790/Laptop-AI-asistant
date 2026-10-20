"""Hybrid retriever (dense + BM25) with product-scope filtering applied at retrieval time.

score = w * cosine + (1 - w) * bm25 / max(bm25)
  - cosine keeps its absolute meaning, so `min_score` on the dense part is used as the
    "no relevant data" signal (BM25 max-normalisation is relative to the query and
    would always give the best chunk 1.0).
  - BM25 rescues exact tokens dense models blur: model numbers, "5070 Ti", "HDMI 2.1".
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aorus_assistant.config import RetrievalConfig
from aorus_assistant.rag.index import BM25, VectorIndex
from aorus_assistant.rag.tokenize import tokenize
from aorus_assistant.schemas import Chunk, RetrievedChunk


# Clause boundaries for multi-part questions: sentence punctuation, enumeration commas, and
# conjunctions between two questions. Only used to *add* context, so over-splitting is cheap.
_SPLIT_RE = re.compile(r"[？?；;。！!、]|，|,|\s+and\s+|\s+&\s+|以及|還有|另外|(?<=[\u4e00-\u9fff])(?:跟|和)(?=[\u4e00-\u9fffA-Za-z])")

# decompose query into several sub-queries
def decompose(query: str, max_parts: int = 3) -> list[str]:
    """Sub-queries of a multi-part question; [] when there is only one part."""
    parts = [p.strip() for p in _SPLIT_RE.split(query)]
    parts = [p for p in parts if len(re.sub(r"\W", "", p)) >= 2]
    return parts[:max_parts] if len(parts) > 1 else []


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run `uv run python scripts/prepare_data.py` and "
            "`uv run python scripts/build_index.py` first (data/processed/ is generated, not committed)"
        )
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

    def score(self, query: str, qvec: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if qvec is None:
            qvec = self.embed_query([query])[0]
        dense = self.index.scores(qvec)
        sparse = self.bm25.scores(tokenize(query))
        return dense, sparse

    def retrieve(
        self, query: str, top_k: int | None = None, qvec: np.ndarray | None = None
    ) -> list[RetrievedChunk]:
        """`qvec`: precomputed query embedding (shared with the input guard)."""
        top_k = top_k or self.cfg.top_k
        dense, sparse = self.score(query, qvec)
        sparse_norm = sparse / sparse.max() if sparse.max() > 0 else sparse
        w = self.cfg.dense_weight
        hybrid = np.where(self.allowed, w * dense + (1 - w) * sparse_norm, -np.inf)

        # candidates: top-k by hybrid score, plus the dense top-k (see select_context)
        dense_masked = np.where(self.allowed, dense, -np.inf)
        order = list(np.argsort(-hybrid)[:top_k])
        order += [i for i in np.argsort(-dense_masked)[: self.cfg.dense_k] if i not in order]
        results = []

        # pack to retrievedChunk
        for i in order:
            if not np.isfinite(hybrid[i]):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[i],
                    score=float(hybrid[i]),
                    dense_score=float(dense[i]),
                    sparse_score=float(sparse_norm[i]),
                )
            )
        return results

    def max_spec_sim(self, qvec: np.ndarray) -> float:
        """Best cosine between the query and any in-scope chunk (product-relevance signal)."""
        scores = self.index.scores(qvec)
        return float(scores[self.allowed].max()) if self.allowed.any() else 0.0

    def select_context(
        self, results: list[RetrievedChunk], sub_best: list[RetrievedChunk] | None = None
    ) -> list[RetrievedChunk]:
        """Union of (a) chunks with hybrid score >= ratio * best, (b) the best chunk of each
        sub-query of a multi-part question, (c) the dense top-k within `dense_margin` of the
        best dense score; in that order, capped at max_context.
        Drops distractors (a 2B model is easily misled) without letting BM25 alone decide."""
        if not results:
            return []
        best = max(r.score for r in results)
        keep = [r for r in results if r.score >= best * self.cfg.context_ratio]
        ids = {r.chunk.id for r in keep}
        for r in sub_best or []:
            if r.chunk.id not in ids:
                keep.append(r)
                ids.add(r.chunk.id)
        by_dense = sorted(results, key=lambda r: -r.dense_score)
        best_dense = by_dense[0].dense_score
        for r in by_dense[: self.cfg.dense_k]:
            if r.dense_score >= best_dense - self.cfg.dense_margin and r.chunk.id not in ids:
                keep.append(r)
                ids.add(r.chunk.id)
        return keep[: self.cfg.max_context]

    def has_relevant(self, results: list[RetrievedChunk]) -> bool:
        return bool(results) and max(r.dense_score for r in results) >= self.cfg.min_score
