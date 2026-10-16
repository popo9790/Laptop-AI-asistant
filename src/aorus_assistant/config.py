"""Central configuration. All tunables live here so benchmarks can record them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

TARGET_PRODUCT = "AORUS MASTER 16 AM6H"
PRODUCT_URLS = {
    "spec_tw": "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp",
    "spec_en": "https://www.gigabyte.com/Laptop/AORUS-MASTER-16-AM6H/sp",
}


@dataclass
class LLMConfig:
    repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all layers
    max_tokens: int = 512
    temperature: float = 0.2


@dataclass
class EmbedConfig:
    repo_id: str = "CompendiumLabs/bge-m3-gguf"
    filename: str = "bge-m3-q8_0.gguf"
    n_gpu_layers: int = 0  # keep embedder on CPU to save VRAM


@dataclass
class RetrievalConfig:
    top_k: int = 5
    dense_weight: float = 0.6  # hybrid score = w * dense + (1 - w) * bm25
    min_score: float = 0.35  # below this -> "no relevant data" refusal


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    moderation_enabled: bool = True
    specs_path: Path = DATA_DIR / "processed" / "specs.json"
    chunks_path: Path = DATA_DIR / "processed" / "chunks.jsonl"
    index_path: Path = DATA_DIR / "processed" / "index.npz"
    brands_path: Path = DATA_DIR / "policy" / "brands.yaml"
    topics_path: Path = DATA_DIR / "policy" / "sensitive_topics.yaml"
    audit_log_path: Path = LOGS_DIR / "audit.jsonl"
