"""Loads the maintainable policy lists (data/policy/*.yaml) and compiles matchers.

All matchers expect text already passed through `normalize()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from aorus_assistant.config import DATA_DIR
from aorus_assistant.moderation.normalize import deleet, normalize

POLICY_DIR = DATA_DIR / "policy"


def _term_regex(terms: list[str]) -> re.Pattern | None:
    """One alternation; latin terms get alphanumeric boundaries so "rog" does not match
    "progress", CJK terms match as substrings."""
    if not terms:
        return None
    parts = []
    for t in sorted({normalize(str(t)) for t in terms}, key=len, reverse=True):  # YAML may give ints
        esc = re.escape(t)
        # optional plural so "monitors" matches "monitor" (v2: v1 missed "Dell UltraSharp monitors")
        parts.append(rf"(?<![a-z0-9]){esc}(?:s|es)?(?![a-z0-9])" if re.search(r"[a-z0-9]", t) else esc)
    return re.compile("|".join(parts))


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


@dataclass
class EntityHit:
    brand: str
    alias: str
    ambiguous: bool


@dataclass
class Lexicon:
    brand_aliases: dict[str, re.Pattern] = field(default_factory=dict)
    brand_ambiguous: dict[str, re.Pattern] = field(default_factory=dict)
    competitor_models: re.Pattern | None = None
    allowed_third_party: re.Pattern | None = None
    product_terms: re.Pattern | None = None
    laptop_terms: re.Pattern | None = None  # hardware_terms + generic_terms
    hardware_terms: re.Pattern | None = None
    own_sku_terms: re.Pattern | None = None
    peripheral_terms: re.Pattern | None = None
    compatibility: list[re.Pattern] = field(default_factory=list)
    comparison: list[re.Pattern] = field(default_factory=list)
    indirect_competitor: list[re.Pattern] = field(default_factory=list)
    topic_keywords: dict[str, re.Pattern] = field(default_factory=dict)
    stance: list[re.Pattern] = field(default_factory=list)
    hate: list[re.Pattern] = field(default_factory=list)
    injection: dict[str, list[re.Pattern]] = field(default_factory=dict)
    prototypes: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------ matching helpers
    def competitor_entities(self, norm: str, require_context: bool = True) -> list[EntityHit]:
        """Competitor brand/model mentions in normalised text (and its leetspeak variant).
        Ambiguous aliases count only with laptop/spec context or another alias of the brand."""
        variants = {norm, deleet(norm)}
        hits: list[EntityHit] = []
        has_laptop_ctx = bool(self.laptop_terms and self.laptop_terms.search(norm))
        for brand, rx in self.brand_aliases.items():
            for v in variants:
                if m := rx.search(v):
                    hits.append(EntityHit(brand, m.group(0), False))
                    break
        if self.competitor_models:
            for v in variants:
                if m := self.competitor_models.search(v):
                    hits.append(EntityHit("model", m.group(0), False))
                    break
        strong_brands = {h.brand for h in hits}
        for brand, rx in self.brand_ambiguous.items():
            for v in variants:
                if (m := rx.search(v)) and (
                    brand in strong_brands or has_laptop_ctx or not require_context
                ):
                    hits.append(EntityHit(brand, m.group(0), True))
                    break
        return hits

    def any_match(self, patterns: list[re.Pattern], norm: str) -> str | None:
        for p in patterns:
            if m := p.search(norm):
                return m.group(0)
        return None

    def has(self, rx: re.Pattern | None, norm: str) -> str | None:
        if rx and (m := rx.search(norm)):
            return m.group(0)
        return None


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_lexicon(policy_dir: Path = POLICY_DIR) -> Lexicon:
    brands = _load_yaml(policy_dir / "brands.yaml")
    scope = _load_yaml(policy_dir / "scope.yaml")
    topics = _load_yaml(policy_dir / "sensitive_topics.yaml")
    injection = _load_yaml(policy_dir / "injection.yaml")
    protos = _load_yaml(policy_dir / "intent_prototypes.yaml")

    lex = Lexicon()
    for brand, spec in brands.get("competitors", {}).items():
        if rx := _term_regex(spec.get("aliases", [])):
            lex.brand_aliases[brand] = rx
        if rx := _term_regex(spec.get("ambiguous", [])):
            lex.brand_ambiguous[brand] = rx
    lex.competitor_models = _term_regex(brands.get("competitor_models", []))
    lex.allowed_third_party = _term_regex(brands.get("allowed_third_party", []))

    lex.product_terms = _term_regex(scope.get("product_terms", []))
    hardware = scope.get("hardware_terms", [])
    lex.hardware_terms = _term_regex(hardware)
    lex.laptop_terms = _term_regex(hardware + scope.get("generic_terms", []))
    lex.own_sku_terms = _term_regex(scope.get("own_sku_terms", []))
    lex.peripheral_terms = _term_regex(scope.get("peripheral_terms", []))
    lex.compatibility = _compile(scope.get("compatibility_patterns", []))
    lex.comparison = _compile(scope.get("comparison_patterns", []))
    lex.indirect_competitor = _compile(scope.get("indirect_competitor_patterns", []))

    for topic, spec in topics.get("topics", {}).items():
        if rx := _term_regex(spec.get("keywords", [])):
            lex.topic_keywords[topic] = rx
    lex.stance = _compile(topics.get("stance_patterns", []))
    lex.hate = _compile(topics.get("hate_patterns", []))

    lex.injection = {k: _compile(v) for k, v in injection.get("patterns", {}).items()}
    lex.prototypes = {k: list(v) for k, v in protos.items()}
    return lex
