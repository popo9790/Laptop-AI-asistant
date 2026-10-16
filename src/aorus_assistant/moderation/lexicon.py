"""Loads maintainable brand / model / alias lists and sensitive-topic definitions from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Lexicon:
    competitor_brands: dict[str, list[str]] = field(default_factory=dict)  # brand -> aliases
    competitor_models: dict[str, list[str]] = field(default_factory=dict)  # model -> aliases
    allowed_third_party: list[str] = field(default_factory=list)  # component vendors / standards
    sensitive_topics: dict[str, dict] = field(default_factory=dict)


def load_lexicon(brands_path: Path, topics_path: Path) -> Lexicon:
    raise NotImplementedError
