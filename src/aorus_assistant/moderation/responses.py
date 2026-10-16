"""Canned refusal / redirect templates per category and language."""

from __future__ import annotations

from aorus_assistant.schemas import Category


def refusal_message(category: Category, lang: str) -> str:
    raise NotImplementedError


def detect_lang(text: str) -> str:
    """Return 'zh' if the text contains CJK characters, else 'en'."""
    raise NotImplementedError
