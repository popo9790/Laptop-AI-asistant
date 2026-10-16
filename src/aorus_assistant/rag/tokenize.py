"""Lightweight tokenizer for BM25 that handles zh-TW / English / mixed text."""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """English/number words + CJK character bigrams."""
    raise NotImplementedError
