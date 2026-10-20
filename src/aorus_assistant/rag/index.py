"""Pure-numpy vector index + BM25."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np


class VectorIndex:
    """Exact cosine search. The corpus is tens of chunks, so brute force beats any ANN
    structure on both latency and recall."""

    def __init__(self, embeddings: np.ndarray, ids: list[str], model_name: str = ""):
        if len(embeddings) != len(ids):
            raise ValueError("embeddings / ids length mismatch")
        self.embeddings = embeddings.astype(np.float32)  # (N, D), L2-normalised
        self.ids = ids
        self.model_name = model_name

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.embeddings @ query_vec.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embeddings=self.embeddings, ids=np.array(self.ids), model=self.model_name)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `uv run python scripts/build_index.py` first")
        data = np.load(path)
        return cls(data["embeddings"], data["ids"].tolist(), str(data["model"]))


class BM25:
    """Okapi BM25 with the Lucene-style non-negative IDF."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        # docs here is tokenized chunk. e.g. [["麥克", "克風", ...], .....]
        self.tfs = [Counter(d) for d in docs]  # tfs is a list with map, e.g. [{"電池":20, "螢幕":2}]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32) # for every chunk, how much words does it have
        self.avgdl = float(self.doc_len.mean()) if docs else 0.0 # avg docs len
        df = Counter(t for d in docs for t in set(d)) # count the num of how many times a word appear in one chunk
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()} # idf for words in chunk (not in query)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32) # the BM25 score for every chunk
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9)) # len penalty
        for t in set(query_tokens): # for every token in query, e.g. ["麥克", "克風", ...]
            idf = self.idf.get(t) # get idf of this token
            if idf is None: # if the token doesn't show in chunk, dont care about it
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=np.float32) # tf of this token in every chunks
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
