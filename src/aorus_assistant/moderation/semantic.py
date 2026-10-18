"""Semantic layer: kNN over labelled intent prototypes (data/policy/intent_prototypes.yaml).

Reuses the retrieval query embedding (same model, same instruction), so the input guard adds
only a (P x D) matrix-vector product on top of retrieval — no second embedding call.
Prototype embeddings are cached in data/processed/ keyed by a hash of the prototype set and
the embedding model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aorus_assistant.config import DATA_DIR


@dataclass
class SemanticResult:
    label: str  # best non-product label, or "product"
    score: float  # mean of top-k similarities for `label`
    product_score: float
    scores: dict[str, float]


class PrototypeClassifier:
    def __init__(
        self,
        prototypes: dict[str, list[str]],
        embed_fn: Callable[[list[str]], np.ndarray],
        model_name: str,
        k: int = 2,
        cache_dir: Path = DATA_DIR / "processed",
    ):
        self.k = k
        self.labels: list[str] = []
        texts: list[str] = []
        for label, items in prototypes.items():
            self.labels += [label] * len(items)
            texts += items
        key = hashlib.sha1(json.dumps([model_name, texts], ensure_ascii=False).encode()).hexdigest()[:12]
        cache = cache_dir / f"prototypes_{key}.npy"
        if cache.exists():
            self.matrix = np.load(cache)
        else:
            self.matrix = embed_fn(texts)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.matrix)
        self.label_idx = {
            lb: np.array([i for i, x in enumerate(self.labels) if x == lb]) for lb in prototypes
        }

    def classify(self, qvec: np.ndarray) -> SemanticResult:
        sims = self.matrix @ qvec
        scores = {}
        for label, idx in self.label_idx.items():
            top = np.sort(sims[idx])[::-1][: self.k]
            scores[label] = float(top.mean())
        product = scores.get("product", 0.0)
        others = {k: v for k, v in scores.items() if k != "product"}
        best = max(others, key=others.get) if others else "product"
        return SemanticResult(best, others.get(best, 0.0), product, scores)
