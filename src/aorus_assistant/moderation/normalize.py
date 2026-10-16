"""Text normalisation to resist evasion: full/half width, case, zero-width chars,
simplified->traditional variants, spacing/punctuation tricks (e.g. "R.O.G", "政 治")."""

from __future__ import annotations


def normalize(text: str) -> str:
    raise NotImplementedError
