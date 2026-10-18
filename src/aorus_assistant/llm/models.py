"""Download GGUF weights and construct llama.cpp handles."""

from __future__ import annotations

from pathlib import Path

from aorus_assistant.config import MODELS_DIR, EmbedConfig, LLMConfig


def ensure_model(repo_id: str, filename: str, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / filename
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, local_dir=models_dir))


def load_llm(cfg: LLMConfig, verbose: bool = False):
    from llama_cpp import Llama

    return Llama(
        model_path=str(ensure_model(cfg.repo_id, cfg.filename)),
        n_ctx=cfg.n_ctx,
        n_batch=cfg.n_batch,
        n_ubatch=cfg.n_ubatch,
        n_gpu_layers=cfg.n_gpu_layers,
        verbose=verbose,
    )


def load_embedder(cfg: EmbedConfig, verbose: bool = False):
    from llama_cpp import Llama

    return Llama(
        model_path=str(ensure_model(cfg.repo_id, cfg.filename)),
        embedding=True,
        n_ctx=cfg.n_ctx,
        n_batch=cfg.n_batch,
        n_ubatch=cfg.n_ubatch,
        n_gpu_layers=cfg.n_gpu_layers,
        verbose=verbose,
    )
