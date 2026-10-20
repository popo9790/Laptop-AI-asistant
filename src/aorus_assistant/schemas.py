"""Shared data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SpecRecord:
    """One spec field of one SKU, e.g. (BZH, 顯示晶片 / Video Graphics) -> RTX 5090 ..."""

    product: str  # "AORUS MASTER 16 AM6H"
    sku: str  # "AORUS MASTER 16 BZH"
    row: int  # field position on the spec page (stable across languages)
    field_zh: str  # "顯示晶片"
    field_en: str  # "Video Graphics"
    lines: list[str]  # value, one entry per line on the page
    notes: list[str]  # footnotes such as "*May vary by scenario"
    source_urls: list[str]


@dataclass
class Chunk:
    id: str # chunk ID
    text: str # the plain text of chunk
    product: str  # used by retrieval filtering
    section: str
    source_url: str
    metadata: dict = field(default_factory=dict) # The real data for LLM


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float # the score after hybrid to sort
    dense_score: float = 0.0 # sematic score
    sparse_score: float = 0.0 # BM25 score


class Category(str, Enum):
    ALLOWED = "allowed"
    SENSITIVE = "sensitive"  # politics / religion / discrimination / hate
    OFF_TOPIC = "off_topic"
    COMPETITOR = "competitor"
    INJECTION = "prompt_injection"
    NO_DATA = "no_relevant_data"


class Action(str, Enum):
    ALLOW = "allow"
    REFUSE = "refuse"
    REDIRECT = "redirect"  # refuse + guide user back to the target product
    FALLBACK = "fallback"  # output replaced by safe response


# This is only for recore every behavier
@dataclass
class ModerationDecision:
    stage: str  # input | retrieval | output
    category: Category
    action: Action
    reason: str
    confidence: float = 1.0
    matched: list[str] = field(default_factory=list)  # rules / terms that fired
    latency_ms: float = 0.0
