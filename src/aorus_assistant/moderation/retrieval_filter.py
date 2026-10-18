"""Stage 2 - Retrieval filtering.

Two layers, so policy is enforced before generation rather than only on the final answer:
  1. Scope pre-filter (`in_scope`): passed to the Retriever as a mask, so chunks whose product
     metadata is not the target product are never scored or returned.
  2. Content check (`filter_chunks`): even in-scope chunks are dropped if their text mentions a
     competitor laptop, contains instruction-like text (prompt injection planted in the knowledge
     base), or contains sensitive-topic content. Protects against a corrupted or poisoned index.
"""

from __future__ import annotations

import time

from aorus_assistant.config import TARGET_PRODUCT
from aorus_assistant.moderation.lexicon import Lexicon
from aorus_assistant.moderation.normalize import normalize
from aorus_assistant.schemas import Action, Category, Chunk, ModerationDecision, RetrievedChunk


def in_scope(chunk: Chunk, product: str = TARGET_PRODUCT) -> bool:
    return chunk.product == product


def chunk_violation(text: str, lex: Lexicon) -> str | None:
    norm = normalize(text)
    for kind, patterns in lex.injection.items():
        if hit := lex.any_match(patterns, norm):
            return f"instruction-like text ({kind}: '{hit}')"
    # Spec chunks always contain laptop vocabulary, so ambiguous aliases need no extra context.
    if ents := lex.competitor_entities(norm, require_context=False):
        return f"competitor mention {[e.alias for e in ents]}"
    for topic, rx in lex.topic_keywords.items():
        if kw := lex.has(rx, norm):
            return f"sensitive content ({topic}: '{kw}')"
    return None


def filter_chunks(
    results: list[RetrievedChunk], lex: Lexicon, product: str = TARGET_PRODUCT
) -> tuple[list[RetrievedChunk], ModerationDecision]:
    t0 = time.perf_counter()
    kept, dropped = [], []
    for rc in results:
        if not in_scope(rc.chunk, product):  # defence in depth if the pre-filter was bypassed
            dropped.append(f"{rc.chunk.id}: out of scope (product='{rc.chunk.product}')")
        elif why := chunk_violation(rc.chunk.text, lex):
            dropped.append(f"{rc.chunk.id}: {why}")
        else:
            kept.append(rc)
    decision = ModerationDecision(
        stage="retrieval",
        category=Category.ALLOWED,
        action=Action.ALLOW,
        reason=f"kept {len(kept)}/{len(results)} chunks" + (f"; dropped {len(dropped)}" if dropped else ""),
        matched=dropped,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return kept, decision
