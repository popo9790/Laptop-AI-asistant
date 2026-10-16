"""Pure-numpy vector index + BM25, persisted as .npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aorus_assistant.schemas import Chunk


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk]):
        self.embeddings = embeddings  # (N, D), L2-normalised
        self.chunks = chunks

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path, chunks: list[Chunk]) -> "VectorIndex":
        raise NotImplementedError


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        raise NotImplementedError

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        raise NotImplementedError
