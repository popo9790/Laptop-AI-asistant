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
    # 2B over 0.8B: on tests/cases/retrieval.jsonl answer accuracy 1.0 vs 0.867 and no-data
    # refusal 1.0 vs 0.556 (0.8B hallucinated prices / part counts); see README.
    repo_id: str = "unsloth/Qwen3.5-2B-GGUF"
    filename: str = "Qwen3.5-2B-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all layers
    # Prompt-processing batch sizes. The GPU compute buffer scales with n_ubatch
    # (worst-case activations for one micro-batch), so it is a VRAM knob.
    n_batch: int = 512
    n_ubatch: int = 512
    max_tokens: int = 512
    temperature: float = 0.0  # greedy: factual QA + reproducible evals
    repeat_penalty: float = 1.1


@dataclass
class EmbedConfig:
    repo_id: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    filename: str = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    n_gpu_layers: int = 0  # keep embedder on CPU to save VRAM
    n_ctx: int = 1024
    n_batch: int = 512
    n_ubatch: int = 512
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
    # Below this top dense score -> "no relevant data" refusal without calling the LLM.
    # v2: 0.35 -> 0.25. Long workload questions with incidental words ("剪總統大選廣告 4K 會不會卡")
    # scored < 0.35 and were refused; Qwen3.5-2B refuses unanswerable questions itself (9/9).
    min_score: float = 0.25
    # Context sent to the LLM: keep chunks scoring >= ratio * best score (max N).
    # A 0.8B model is easily distracted; on the retrieval set relevant chunks sit at
    # >= 0.8 of the top score and distractors at <= 0.7.
    context_ratio: float = 0.75
    max_context: int = 4
    # Also keep the top-`dense_k` chunks by dense score when within `dense_margin` of the best
    # dense score. BM25 max-normalisation gives its favourite chunk 1.0, so on its own the
    # ratio cutoff dropped chunks dense ranked first ("寬跟深幾公分" kept only Weight because
    # 公分/公斤 share 公) and the second field of two-part questions ("麥克風跟 webcam").
    dense_k: int = 2
    dense_margin: float = 0.05
    # Multi-part questions ("鏡頭解析度？喇叭配置？", "width and weight") are split into
    # sub-queries; each sub-query's best chunk joins the context (one extra embedding each).
    max_subqueries: int = 3


@dataclass
class ModerationConfig:
    # Semantic layer (kNN over intent prototypes, reusing the retrieval query embedding):
    # flag the best non-product label when its mean top-2 cosine similarity is >= a floor AND
    # beats the product-prototype similarity by >= margin.
    # Calibrated on own data only (39 product questions in retrieval.jsonl + prototype
    # leave-one-out), never on the blind test set:
    #   margin 0.05 -> prototype LOO 68/88, own product questions flagged 1/39
    #   margin 0.10 -> prototype LOO 60/88, own product questions flagged 0/39   <- chosen
    # A task-specific "classify intent" instruction did not separate better (LOO 62/88 @0.05)
    # and would cost a second embedding call, so the retrieval embedding is reused.
    sem_threshold: dict = field(
        default_factory=lambda: {
            "competitor": 0.45,
            "politics": 0.45,
            "religion": 0.45,
            "discrimination_hate": 0.45,
            "off_topic": 0.45,
            "injection": 0.45,
        }
    )
    sem_margin: float = 0.10
    # Off-topic fallback: no product/laptop vocabulary and product similarity below this.
    offtopic_product_floor: float = 0.45
    # Product signal from retrieval: best cosine to any in-scope spec chunk >= this exempts a
    # query from the vocabulary-based off-topic fallback (v3: vocabulary alone missed
    # "WD Black SN850X 裝得進去嗎"). v3.1: not used for the sensitive gate or to override a
    # semantic off-topic match (that let "台獨是對的嗎" and "嗨～你今天過得怎麼樣" through).
    # Own data: product questions >= 0.45 in 85%, off-topic/sensitive prototypes in 5%.
    product_spec_sim: float = 0.45
    # Output guard: characters held back from the stream until checked.
    holdback_chars: int = 16


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    moderation: ModerationConfig = field(default_factory=ModerationConfig)
    moderation_enabled: bool = True
    specs_path: Path = DATA_DIR / "processed" / "specs.json"
    chunks_path: Path = DATA_DIR / "processed" / "chunks.jsonl"
    index_path: Path = DATA_DIR / "processed" / "index.npz"
    brands_path: Path = DATA_DIR / "policy" / "brands.yaml"
    topics_path: Path = DATA_DIR / "policy" / "sensitive_topics.yaml"
    audit_log_path: Path = LOGS_DIR / "audit.jsonl"
