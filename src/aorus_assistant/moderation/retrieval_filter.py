"""Stage 2 - Retrieval filtering: product-scope metadata filter + content checks on chunks
(drops competitor / policy-violating / instruction-like chunks before they reach the LLM)."""

from __future__ import annotations

from aorus_assistant.moderation.lexicon import Lexicon
from aorus_assistant.schemas import Chunk, ModerationDecision, RetrievedChunk


def in_scope(chunk: Chunk, product: str) -> bool:
    """Pre-filter passed to the Retriever so out-of-scope chunks are never scored."""
    raise NotImplementedError


def filter_chunks(
    chunks: list[RetrievedChunk], lexicon: Lexicon
) -> tuple[list[RetrievedChunk], ModerationDecision]:
    raise NotImplementedError
