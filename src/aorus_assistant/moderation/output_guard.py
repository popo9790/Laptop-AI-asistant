"""Stage 4 - Output moderation: competitor mentions, sensitive content, ungrounded numbers.
Works on the streamed buffer so it can cut off a response mid-stream."""

from __future__ import annotations

from aorus_assistant.moderation.lexicon import Lexicon
from aorus_assistant.schemas import ModerationDecision, RetrievedChunk


class OutputGuard:
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon

    def check(self, answer: str, contexts: list[RetrievedChunk]) -> ModerationDecision:
        raise NotImplementedError
