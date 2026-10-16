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
    repo_id: str = "unsloth/Qwen3.5-0.8B-GGUF"
    filename: str = "Qwen3.5-0.8B-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all layers
    max_tokens: int = 512
    temperature: float = 0.2


@dataclass
class EmbedConfig:
    repo_id: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    filename: str = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    n_gpu_layers: int = 0  # keep embedder on CPU to save VRAM
    n_ctx: int = 1024
    # Asymmetric retrieval: Qwen3-Embedding expects an instruction on the query side only.
    # Set to "" for symmetric models such as bge-m3.
    query_prefix: str = (
        "Instruct: Given a user question about a laptop, retrieve the matching "
        "specification entry\nQuery: "
    )
    doc_prefix: str = ""


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
