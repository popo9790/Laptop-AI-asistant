"""Download GGUF weights and construct llama.cpp handles."""

from __future__ import annotations

from pathlib import Path

from aorus_assistant.config import MODELS_DIR, EmbedConfig, LLMConfig


def ensure_model(repo_id: str, filename: str, models_dir: Path = MODELS_DIR) -> Path:
    raise NotImplementedError


def load_llm(cfg: LLMConfig):
    raise NotImplementedError


def load_embedder(cfg: EmbedConfig):
    raise NotImplementedError
