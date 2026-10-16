"""Stage 1 - Input moderation (before retrieval).

Hybrid design:
  1. normalise text
  2. rule layer: injection patterns, competitor lexicon, sensitive-topic lexicon
  3. semantic layer: embedding similarity to labelled intent prototypes
     (catches paraphrases / indirect competitor queries)
  4. allow-list overrides for legitimate third-party / compatibility questions
"""

from __future__ import annotations

from aorus_assistant.moderation.lexicon import Lexicon
from aorus_assistant.schemas import ModerationDecision


class InputGuard:
    def __init__(self, lexicon: Lexicon, embed_fn=None):
        self.lexicon = lexicon
        self.embed_fn = embed_fn

    def check(self, query: str) -> ModerationDecision:
        raise NotImplementedError
