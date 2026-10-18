"""Dense embeddings via llama.cpp (GGUF embedding model, CPU by default)."""

from __future__ import annotations

import numpy as np

from aorus_assistant.config import EmbedConfig
from aorus_assistant.llm.models import load_embedder


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class Embedder:
    def __init__(self, cfg: EmbedConfig, model=None, verbose: bool = False):
        self.cfg = cfg
        self.model = model if model is not None else load_embedder(cfg, verbose=verbose)

    def _embed(self, texts: list[str]) -> np.ndarray:
        # One call per text: llama.cpp pools per sequence, and batching mixed-length
        # inputs gains little on CPU for our corpus size.
        vecs = [self.model.embed(t) for t in texts]
        return l2_normalize(np.asarray(vecs, dtype=np.float32))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.cfg.doc_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.cfg.query_prefix + t for t in texts])
