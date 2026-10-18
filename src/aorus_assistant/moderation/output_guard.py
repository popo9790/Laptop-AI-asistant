"""Stage 4 - Output moderation on the streamed answer.

Checks (on normalised text):
  - competitor laptop brand/model mentioned (unless the user's own question legitimately named
    that brand as a peripheral vendor and the input guard allowed it)       -> full fallback
  - sensitive-topic keywords not already present in the question             -> full fallback
  - ungrounded numbers: a numeric claim that appears neither in the retrieved context nor in
    the question (hallucination guard: invented prices, part counts, noise levels)
                                                                             -> clause redacted

Streaming: text is released clause by clause (，、：；。！？ or newline), each clause checked
before release, so a brand name or invented number never reaches the user. Competitor or
sensitive content stops generation and replaces the answer. An ungrounded number removes only
its clause; generation continues and a notice is appended.
History: v1 replaced the whole answer (threw away correct sentences such as "重量約 2.5 kg");
v2 truncated at the first ungrounded number (a bad first clause left "相關資訊如下：（省略）");
v3 redacts clauses.
Numbers inside identifiers ("H.265", "DDR4", "USB3.2") and list numbering are not claims.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from aorus_assistant.moderation.lexicon import Lexicon
from aorus_assistant.moderation.normalize import normalize
from aorus_assistant.schemas import Action, Category, ModerationDecision, RetrievedChunk

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
# numeric claims: not glued to a preceding letter ("H.265", "DDR4", "USB3.2" are names)
_CLAIM_NUM_RE = re.compile(r"(?<![A-Za-z\d.,])(?<![A-Za-z][.\-])\d+(?:[.,]\d+)*")
_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*(?:[#>*\-]+[ \t]*)*(?:\*\*)?\d{1,2}(?:\*\*)?[.)、:：]")
_CLAUSE_END_RE = re.compile(r"[，、；;：。！？!?\n]|,(?=\s)|:(?=\s)|\.(?=\s)")
REDACTION_NOTICE = {
    "zh": "（部分超出產品規格資料範圍的內容已省略。）",
    "en": "(Some content beyond the product specification data was omitted.)",
}


def _norm_num(raw: str) -> str:
    n = raw.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def numbers(text: str) -> set[str]:
    """All numbers, normalised: '1,000,000' -> '1000000', '2.50' -> '2.5'."""
    return {_norm_num(x) for x in _NUM_RE.findall(text)}


def claim_numbers(text: str) -> set[str]:
    """Numbers that state a value (excludes list numbering and identifier-embedded digits)."""
    body = _LIST_MARKER_RE.sub(lambda m: " " * len(m.group(0)), text)
    return {_norm_num(x) for x in _CLAIM_NUM_RE.findall(body)}


def context_text(contexts: list[RetrievedChunk]) -> str:
    parts = []
    for rc in contexts:
        meta = rc.chunk.metadata
        parts.append(rc.chunk.text)
        parts.extend(meta.get("lines", []))
        parts.extend(meta.get("notes", []))
    return "\n".join(parts)


@dataclass
class OutputGuard:
    lex: Lexicon
    holdback: int = 16  # kept for config compatibility; v3 releases per clause

    def check_policy(
        self, answer: str, query: str, allowed_brands: set[str] | None = None
    ) -> ModerationDecision | None:
        """Competitor / sensitive content anywhere in the answer -> violation."""
        norm = normalize(answer)
        allowed_brands = allowed_brands or set()
        ents = [e for e in self.lex.competitor_entities(norm) if e.brand not in allowed_brands]
        if ents:
            return self._violation(Category.COMPETITOR, f"answer mentions competitor {[e.alias for e in ents]}")
        qnorm = normalize(query)
        for topic, rx in self.lex.topic_keywords.items():
            kw = self.lex.has(rx, norm)
            if kw and kw not in qnorm:
                return self._violation(Category.SENSITIVE, f"answer contains {topic} term '{kw}'")
        return None

    def ungrounded(self, text: str, contexts: list[RetrievedChunk], query: str) -> list[str]:
        grounded = numbers(context_text(contexts)) | numbers(query)
        return sorted(claim_numbers(text) - grounded, key=len, reverse=True)

    def check(
        self, answer: str, contexts: list[RetrievedChunk], query: str,
        allowed_brands: set[str] | None = None,
    ) -> ModerationDecision | None:
        """Whole-answer check (non-streaming use and tests)."""
        if v := self.check_policy(answer, query, allowed_brands):
            return v
        if bad := self.ungrounded(answer, contexts, query):
            return self._violation(Category.NO_DATA, f"ungrounded number(s) {bad[:5]}")
        return None

    @staticmethod
    def _violation(category: Category, reason: str) -> ModerationDecision:
        return ModerationDecision(stage="output", category=category, action=Action.FALLBACK, reason=reason)


@dataclass
class GuardedStream:
    """Wraps a delta iterator; after exhaustion `decision` holds the output-stage decision and
    `redacted` the removed clauses."""

    guard: OutputGuard
    stream: Iterator[str]
    contexts: list[RetrievedChunk]
    query: str
    fallback: str
    allowed_brands: set[str] = field(default_factory=set)
    lang: str = "zh"
    decision: ModerationDecision | None = None
    check_ms: float = 0.0
    raw_text: str = ""
    released: str = ""
    redacted: list[tuple[str, list[str]]] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        pending = ""
        for delta in self.stream:
            self.raw_text += delta
            pending += delta
            if v := self._timed(self.guard.check_policy, self.raw_text, self.query, self.allowed_brands):
                yield from self._fallback(v)
                return
            while m := _CLAUSE_END_RE.search(pending):
                clause, pending = pending[: m.end()], pending[m.end():]
                yield from self._release(clause)
        if pending:
            yield from self._release(pending)

        if self.redacted and not self.released.strip():
            v = OutputGuard._violation(
                Category.NO_DATA, f"every clause had ungrounded numbers {[b for _, b in self.redacted]}")
            yield from self._fallback(v)
            return
        if self.redacted:
            yield "\n" + REDACTION_NOTICE[self.lang]
        reason = "output passed checks"
        if self.redacted:
            reason = f"redacted {len(self.redacted)} clause(s) with ungrounded numbers " + str(
                [(c[:30], b) for c, b in self.redacted])
        self.decision = ModerationDecision(
            stage="output", category=Category.ALLOWED, action=Action.ALLOW, reason=reason,
            latency_ms=self.check_ms,
        )

    def _timed(self, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        self.check_ms += (time.perf_counter() - t0) * 1000
        return out

    def _release(self, clause: str) -> Iterator[str]:
        bad = self._timed(self.guard.ungrounded, clause, self.contexts, self.query)
        if bad:
            self.redacted.append((clause.strip(), bad))
            return
        self.released += clause
        yield clause

    def _fallback(self, violation: ModerationDecision) -> Iterator[str]:
        """Stop generation (closing the generator stops llama.cpp) and show the fallback."""
        violation.latency_ms = self.check_ms
        violation.matched = [f"released_chars:{len(self.released)}"]
        self.decision = violation
        close = getattr(self.stream, "close", None)
        if close:
            close()
        yield ("\n\n" if self.released else "") + self.fallback
