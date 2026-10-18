"""Stage 1 - Input moderation (before retrieval).

Hybrid design, cheapest and most precise checks first:
  1. normalise (normalize.py)
  2. injection rules (+ base64 / ROT13 decoded payloads)
  3. competitor rules: brand/model/alias entities gated by intent
       - compatibility intent + peripheral noun, no comparison -> allowed third-party mention
       - otherwise -> competitor
     indirect competitor phrases without a brand name -> competitor
  4. sensitive keywords gated by stance / hate markers and product context
  5. semantic kNN over intent prototypes (paraphrases, indirect intent, no keywords)
  6. off-topic fallback: no product vocabulary and low similarity to product prototypes
Every decision records which rule fired, for audit and error analysis.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import time

import numpy as np

from aorus_assistant.config import ModerationConfig
from aorus_assistant.moderation.lexicon import Lexicon
from aorus_assistant.moderation.normalize import normalize
from aorus_assistant.moderation.semantic import PrototypeClassifier
from aorus_assistant.schemas import Action, Category, ModerationDecision

SEM_LABEL_TO_CATEGORY = {
    "competitor": Category.COMPETITOR,
    "politics": Category.SENSITIVE,
    "religion": Category.SENSITIVE,
    "discrimination_hate": Category.SENSITIVE,
    "off_topic": Category.OFF_TOPIC,
    "injection": Category.INJECTION,
}
ACTION = {
    Category.COMPETITOR: Action.REDIRECT,
    Category.OFF_TOPIC: Action.REDIRECT,
    Category.SENSITIVE: Action.REFUSE,
    Category.INJECTION: Action.REFUSE,
}
_B64_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_LATIN_WORDS_RE = re.compile(r"[a-z]{3,}(?:\s+[a-z]{2,}){3,}")


class InputGuard:
    def __init__(
        self,
        lexicon: Lexicon,
        cfg: ModerationConfig,
        classifier: PrototypeClassifier | None = None,
    ):
        self.lex = lexicon
        self.cfg = cfg
        self.clf = classifier

    # ------------------------------------------------------------------ rule layers
    def _injection(self, norm: str) -> tuple[str, str] | None:
        for kind, patterns in self.lex.injection.items():
            if hit := self.lex.any_match(patterns, norm):
                return kind, hit
        return None

    def _decoded_payloads(self, raw: str) -> list[tuple[str, str]]:
        """Base64 tokens and ROT13 of long latin runs, normalised for re-scanning."""
        out = []
        for tok in _B64_RE.findall(raw):
            try:
                text = base64.b64decode(tok + "=" * (-len(tok) % 4), validate=True).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            if text.isprintable():
                out.append(("base64", normalize(text)))
        for run in _LATIN_WORDS_RE.findall(raw.lower()):
            out.append(("rot13", normalize(codecs.decode(run, "rot13"))))
        return out

    def _competitor(self, norm: str) -> tuple[ModerationDecision | None, list[str]]:
        notes: list[str] = []
        entities = self.lex.competitor_entities(norm)
        if entities:
            aliases = [e.alias for e in entities]
            compare = self.lex.any_match(self.lex.comparison, norm)
            compat = self.lex.any_match(self.lex.compatibility, norm)
            peripheral = self.lex.has(self.lex.peripheral_terms, norm)
            if compat and peripheral and not compare:
                notes.append(f"third_party_peripheral:{aliases}")
            else:
                why = f"comparison '{compare}'" if compare else "competitor product request"
                return self._decision(
                    Category.COMPETITOR, f"competitor entity {aliases}; {why}",
                    [f"entity:{a}" for a in aliases] + ([f"comparison:{compare}"] if compare else []),
                ), notes
        if hit := self.lex.any_match(self.lex.indirect_competitor, norm):
            return self._decision(
                Category.COMPETITOR, f"indirect competitor intent '{hit}'", [f"indirect:{hit}"]
            ), notes
        return None, notes

    def _sensitive(self, norm: str, spec_sim: float | None) -> tuple[ModerationDecision | None, list[str]]:
        notes: list[str] = []
        # Lexical product context only: a sensitive keyword plus mere embedding similarity to the
        # spec ("台獨是對的嗎" scored >= 0.45) must not pass (v3.1 regression fix).
        product_ctx = self._product_ctx(norm)
        for topic, rx in self.lex.topic_keywords.items():
            kw = self.lex.has(rx, norm)
            if not kw:
                continue
            stance = self.lex.any_match(self.lex.stance, norm)
            hate = self.lex.any_match(self.lex.hate, norm) if topic == "discrimination_hate" else None
            if stance or hate or not product_ctx:
                marker = f"stance '{stance}'" if stance else f"hate '{hate}'" if hate else "no product context"
                return self._decision(
                    Category.SENSITIVE, f"{topic} keyword '{kw}' with {marker}",
                    [f"{topic}:{kw}"] + ([f"stance:{stance}"] if stance else []) + ([f"hate:{hate}"] if hate else []),
                ), notes
            notes.append(f"sensitive_term_in_product_context:{topic}:{kw}")
        return None, notes

    def _product_ctx(self, norm: str) -> str | None:
        """Lexical product context: product / hardware vocabulary. Generic words ("model",
        "upgrade") are excluded: v1 let "Tesla Model Y" and "upgrading my PS5" skip the
        off-topic check. Embedding similarity to the spec (`spec_sim`) is used separately and
        only for the off-topic fallback."""
        return self.lex.has(self.lex.product_terms, norm) or self.lex.has(self.lex.hardware_terms, norm)

    # ------------------------------------------------------------------ main entry
    def check(
        self, query: str, qvec: np.ndarray | None = None, spec_sim: float | None = None
    ) -> ModerationDecision:
        t0 = time.perf_counter()
        decision = self._check(query, qvec, spec_sim)
        decision.latency_ms = (time.perf_counter() - t0) * 1000
        return decision

    def _check(self, query: str, qvec: np.ndarray | None, spec_sim: float | None) -> ModerationDecision:
        norm = normalize(query)
        notes: list[str] = []

        if inj := self._injection(norm):
            return self._decision(Category.INJECTION, f"injection rule '{inj[0]}': '{inj[1]}'", [f"injection:{inj[0]}"])
        for enc, decoded in self._decoded_payloads(query):
            if inj := self._injection(decoded):
                return self._decision(
                    Category.INJECTION, f"{enc}-encoded injection '{inj[0]}'", [f"injection:{inj[0]}", f"encoded:{enc}"]
                )

        dec, n = self._competitor(norm)
        notes += n
        if dec:
            return dec
        dec, n = self._sensitive(norm, spec_sim)
        notes += n
        if dec:
            return dec

        product_ctx = self._product_ctx(norm)
        own_sku = self.lex.has(self.lex.own_sku_terms, norm)
        if qvec is not None and self.clf is not None:
            sem = self.clf.classify(qvec)
            thr = self.cfg.sem_threshold.get(sem.label, 1.0)
            margin = sem.score - sem.product_score
            # Sensitive-term-in-product-context passes were deliberate; don't let the
            # semantic layer re-block them on topic similarity alone.
            sensitive_pass = any(x.startswith("sensitive_term_in_product_context") for x in notes)
            third_party_pass = any(x.startswith("third_party_peripheral") for x in notes)
            category = SEM_LABEL_TO_CATEGORY.get(sem.label)
            skip = (category == Category.SENSITIVE and sensitive_pass) or (
                category == Category.COMPETITOR and third_party_pass
            ) or (category == Category.OFF_TOPIC and product_ctx) or (
                # comparisons among this product's own SKUs look like comparisons to the
                # embedding (v1: "RTX 5090 版跟 5080 版差在哪" -> competitor); the rule layer
                # already checked for competitor entities and indirect phrases
                category == Category.COMPETITOR and own_sku
            )
            if category and not skip and sem.score >= thr and margin >= self.cfg.sem_margin:
                return self._decision(
                    category,
                    f"semantic '{sem.label}' sim={sem.score:.2f} (product {sem.product_score:.2f})",
                    [f"semantic:{sem.label}"],
                    confidence=sem.score,
                    notes=notes,
                )
            # spec_sim only rescues questions from this vocabulary-based fallback (no semantic
            # evidence of anything else), e.g. "WD Black SN850X 4TB 裝得進去嗎"
            spec_ctx = spec_sim is not None and spec_sim >= self.cfg.product_spec_sim
            if not product_ctx and not spec_ctx and sem.product_score < self.cfg.offtopic_product_floor:
                return self._decision(
                    Category.OFF_TOPIC,
                    f"no product vocabulary and product similarity {sem.product_score:.2f} "
                    f"< {self.cfg.offtopic_product_floor}",
                    ["offtopic:fallback"],
                    confidence=1 - sem.product_score,
                    notes=notes,
                )
            notes.append(f"semantic_best:{sem.label}:{sem.score:.2f}/product:{sem.product_score:.2f}")

        return ModerationDecision(
            stage="input", category=Category.ALLOWED, action=Action.ALLOW,
            reason="no policy rule or semantic match", matched=notes,
        )

    @staticmethod
    def _decision(category, reason, matched, confidence=1.0, notes=None) -> ModerationDecision:
        return ModerationDecision(
            stage="input", category=category, action=ACTION[category], reason=reason,
            confidence=confidence, matched=matched + (notes or []),
        )
