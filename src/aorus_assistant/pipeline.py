"""End-to-end pipeline:
input guard -> retrieval (scope-filtered) -> retrieval filter -> grounded generation (stream)
-> output guard -> audit log.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from aorus_assistant.config import Config
from aorus_assistant.schemas import ModerationDecision, RetrievedChunk


@dataclass
class PipelineResult:
    answer: str = ""
    refused: bool = False
    decisions: list[ModerationDecision] = field(default_factory=list)
    contexts: list[RetrievedChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class AssistantPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

    def stream(self, query: str) -> Iterator[str]:
        raise NotImplementedError

    def run(self, query: str) -> PipelineResult:
        raise NotImplementedError
