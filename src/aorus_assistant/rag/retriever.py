"""Hybrid retriever (dense + BM25) with product-scope filtering applied at retrieval time."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aorus_assistant.config import RetrievalConfig
from aorus_assistant.schemas import Chunk, RetrievedChunk


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embed_fn: Callable[[list[str]], np.ndarray],
        cfg: RetrievalConfig,
        chunk_filter: Callable[[Chunk], bool] | None = None,
    ):
        self.chunks = chunks
        self.embed_fn = embed_fn
        self.cfg = cfg
        self.chunk_filter = chunk_filter

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        raise NotImplementedError
