"""Canned refusal / redirect templates per category and language."""

from __future__ import annotations

import re

from aorus_assistant.schemas import Category

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def refusal_message(category: Category, lang: str) -> str:
    raise NotImplementedError


NO_DATA_TEXT = {
    "zh": "目前的產品資料中沒有這項資訊。您可以詢問 AORUS MASTER 16 AM6H 的處理器、顯示卡、螢幕、記憶體、連接埠、電池等規格。",
    "en": "I don't have that information in the product data. You can ask about the AORUS MASTER 16 AM6H's CPU, GPU, display, memory, ports, battery and other specs.",
}


def refusal_text(lang: str) -> str:
    """Fallback when retrieval finds nothing relevant (no LLM call)."""
    return NO_DATA_TEXT[lang]


def detect_lang(text: str) -> str:
    """Return 'zh' if the text contains CJK characters, else 'en'.
    Mixed zh/en queries are answered in Chinese (target users are zh-TW)."""
    return "zh" if _CJK_RE.search(text) else "en"
