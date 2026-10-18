"""End-to-end pipeline with moderation at every stage:

  query
   -> embed once (shared by input guard and retrieval)
   -> [1] input guard ............ refuse / redirect early, no retrieval or LLM call
   -> retrieval (scope pre-filter: only target-product chunks are scored)
   -> [2] retrieval filter ....... drop competitor / instruction-like / sensitive chunks
   -> no-data threshold .......... refuse without calling the LLM
   -> [3] generation guard ....... grounded system prompt, <context> isolation
   -> [4] output guard ........... hold-back streaming check, fallback on violation
   -> [5] audit log

`moderation=False` gives the baseline used for before/after benchmarks: no input guard,
no scope pre-filter, no retrieval filter, no output guard (the system prompt is unchanged).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from aorus_assistant.bench.timing import GenStats, measure_stream
from aorus_assistant.config import Config
from aorus_assistant.llm.generator import stream_answer
from aorus_assistant.llm.models import load_llm
from aorus_assistant.moderation.audit import AuditLogger
from aorus_assistant.moderation.input_guard import InputGuard
from aorus_assistant.moderation.lexicon import load_lexicon
from aorus_assistant.moderation.normalize import normalize
from aorus_assistant.moderation.output_guard import GuardedStream, OutputGuard
from aorus_assistant.moderation.responses import detect_lang, refusal_message
from aorus_assistant.moderation.retrieval_filter import filter_chunks, in_scope
from aorus_assistant.moderation.semantic import PrototypeClassifier
from aorus_assistant.rag.embedder import Embedder
from aorus_assistant.rag.index import VectorIndex
from aorus_assistant.rag.retriever import Retriever, decompose, load_chunks
from aorus_assistant.schemas import Category, Chunk, ModerationDecision, RetrievedChunk


@dataclass
class PipelineResult:
    request_id: str = ""
    query: str = ""
    answer: str = ""
    refused: bool = False
    refusal_stage: str | None = None  # input | no_data | output
    category: Category = Category.ALLOWED
    decisions: list[ModerationDecision] = field(default_factory=list)
    contexts: list[RetrievedChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    n_tokens: int = 0
    decode_tps: float | None = None
    redacted: bool = False  # output guard removed clauses with ungrounded numbers


class AssistantPipeline:
    def __init__(
        self,
        cfg: Config | None = None,
        moderation: bool | None = None,
        llm=None,
        embedder: Embedder | None = None,
        extra_chunks: list[tuple[Chunk, np.ndarray]] | None = None,
        audit: bool = True,
    ):
        """`extra_chunks`: (chunk, embedding) pairs appended to the index, used by the
        adversarial evaluation to simulate a poisoned knowledge base."""
        self.cfg = cfg or Config()
        self.moderation = self.cfg.moderation_enabled if moderation is None else moderation
        self.embedder = embedder or Embedder(self.cfg.embed)
        self.llm = llm or load_llm(self.cfg.llm)
        self.lex = load_lexicon()

        chunks = load_chunks(self.cfg.chunks_path)
        index = VectorIndex.load(self.cfg.index_path)
        if extra_chunks:
            chunks = chunks + [c for c, _ in extra_chunks]
            index = VectorIndex(
                np.vstack([index.embeddings, *[v[None, :] for _, v in extra_chunks]]),
                [c.id for c in chunks],
                index.model_name,
            )
        self.retriever = Retriever(
            chunks, index, self.embedder.embed_queries, self.cfg.retrieval,
            chunk_filter=in_scope if self.moderation else None,
        )
        classifier = PrototypeClassifier(
            self.lex.prototypes, self.embedder.embed_queries, self.cfg.embed.filename
        )
        self.input_guard = InputGuard(self.lex, self.cfg.moderation, classifier)
        self.output_guard = OutputGuard(self.lex, self.cfg.moderation.holdback_chars)
        self.audit = AuditLogger(self.cfg.audit_log_path) if audit else None
        self._count = lambda t: len(self.llm.tokenize(t.encode(), add_bos=False, special=False))

    def answer(self, query: str, on_token: Callable[[str], None] | None = None) -> PipelineResult:
        """Run the pipeline. Text is streamed through `on_token` as it is released;
        timings include `ttft` (request start -> first released text)."""
        res = PipelineResult(request_id=uuid.uuid4().hex[:12], query=query)
        t_start = time.perf_counter()
        first: list[float] = []
        parts: list[str] = []

        def emit(text: str) -> None:
            if not text:
                return
            if not first:
                first.append(time.perf_counter())
            parts.append(text)
            if on_token:
                on_token(text)

        lang = detect_lang(query)
        t = time.perf_counter()
        qvec = self.embedder.embed_queries([query])[0]
        res.timings_ms["embed"] = (time.perf_counter() - t) * 1000

        if self.moderation:
            spec_sim = self.retriever.max_spec_sim(qvec)
            dec = self.input_guard.check(query, qvec, spec_sim)
            res.decisions.append(dec)
            res.timings_ms["input_guard"] = dec.latency_ms
            if dec.category != Category.ALLOWED:
                return self._finish(res, refusal_message(dec.category, lang), emit, parts, first,
                                    t_start, "input", dec.category)

        contexts, dec = self.build_context(query, qvec, res.timings_ms)
        if dec:
            res.decisions.append(dec)
        if not contexts:
            return self._finish(res, refusal_message(Category.NO_DATA, lang), emit, parts, first,
                                t_start, "no_data", Category.NO_DATA)
        res.contexts = contexts
        llm_cfg = self.cfg.llm
        gen = GenStats()
        stream = measure_stream(
            stream_answer(self.llm, query, res.contexts, lang, llm_cfg.max_tokens,
                          llm_cfg.temperature, llm_cfg.repeat_penalty),
            gen, self._count,
        )
        t = time.perf_counter()
        if self.moderation:
            allowed = self._allowed_brands(query, res.decisions[0])
            guarded = GuardedStream(self.output_guard, stream, res.contexts, query,
                                    refusal_message(Category.NO_DATA, lang), allowed, lang)
            for delta in guarded:
                emit(delta)
            res.decisions.append(guarded.decision)
            res.timings_ms["output_guard"] = guarded.check_ms
            raw = guarded.raw_text
            res.redacted = bool(guarded.redacted)
            if guarded.decision.category != Category.ALLOWED:
                res.refused, res.refusal_stage = True, "output"
                res.category = guarded.decision.category
        else:
            raw = ""
            for delta in stream:
                raw += delta
                emit(delta)
        res.timings_ms["generation"] = (time.perf_counter() - t) * 1000
        # LLM-only numbers (exclude moderation hold-back): prefill latency and decode speed
        res.timings_ms["llm_ttft"] = gen.ttft_ms
        res.decode_tps = gen.decode_tps
        res.n_tokens = gen.n_tokens or self._count(raw)
        return self._finish(res, None, emit, parts, first, t_start)

    def build_context(
        self, query: str, qvec: np.ndarray | None = None, timings: dict | None = None
    ) -> tuple[list[RetrievedChunk], ModerationDecision | None]:
        """Retrieval -> [retrieval filter] -> sub-query expansion -> context selection.
        Returns ([], decision) when nothing relevant survives (no-data refusal)."""
        timings = timings if timings is not None else {}
        t = time.perf_counter()
        results = self.retriever.retrieve(query, qvec=qvec)
        sub_results = [self.retriever.retrieve(sq, top_k=1)
                       for sq in decompose(query, self.cfg.retrieval.max_subqueries)]
        timings["retrieval"] = (time.perf_counter() - t) * 1000
        dec = None
        if self.moderation:
            results, dec = filter_chunks(results, self.lex)
            filtered = [filter_chunks(r, self.lex) for r in sub_results]
            sub_results = [r for r, _ in filtered]
            dec.matched += [m for _, d in filtered for m in d.matched]
            dec.latency_ms += sum(d.latency_ms for _, d in filtered)
            timings["retrieval_filter"] = dec.latency_ms
        if not self.retriever.has_relevant(results):
            return [], dec
        sub_best = [r[0] for r in sub_results if r and r[0].dense_score >= self.cfg.retrieval.min_score]
        return self.retriever.select_context(results, sub_best), dec

    def _allowed_brands(self, query: str, input_decision: ModerationDecision) -> set[str]:
        """Brands the user legitimately named as peripheral vendors may appear in the answer."""
        if not any(m.startswith("third_party_peripheral") for m in input_decision.matched):
            return set()
        return {e.brand for e in self.lex.competitor_entities(normalize(query), False)}

    def _finish(self, res, refusal, emit, parts, first, t_start, stage=None, category=None):
        if refusal is not None:
            emit(refusal)
            res.refused, res.refusal_stage, res.category = True, stage, category
        res.answer = "".join(parts)
        res.timings_ms["total"] = (time.perf_counter() - t_start) * 1000
        if first:
            res.timings_ms["ttft"] = (first[0] - t_start) * 1000
        if self.audit:
            self.audit.log(
                res.request_id, res.query, res.decisions,
                moderation=self.moderation, refused=res.refused, refusal_stage=res.refusal_stage,
                redacted=res.redacted,
                category=res.category.value, contexts=[rc.chunk.id for rc in res.contexts],
                timings_ms={k: round(v, 2) for k, v in res.timings_ms.items()},
            )
        return res
