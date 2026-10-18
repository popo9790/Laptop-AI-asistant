"""Canned refusal / redirect templates per category and language."""

from __future__ import annotations

import re

from aorus_assistant.schemas import Category

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


_SUGGEST = {
    "zh": "您可以詢問 AORUS MASTER 16 AM6H 的處理器、顯示卡、螢幕、記憶體、連接埠、電池等規格。",
    "en": "You can ask about the AORUS MASTER 16 AM6H's CPU, GPU, display, memory, ports, battery and other specs.",
}
REFUSALS = {
    Category.SENSITIVE: {
        "zh": "抱歉，我是筆電規格小幫手，無法討論政治、宗教或涉及族群的議題。",
        "en": "Sorry, I'm a laptop spec assistant and can't discuss political, religious, or group-related topics.",
    },
    Category.OFF_TOPIC: {
        "zh": "抱歉，這個問題超出我的服務範圍，我只能回答 AORUS MASTER 16 AM6H 的產品問題。",
        "en": "Sorry, that's outside what I can help with. I only answer questions about the AORUS MASTER 16 AM6H.",
    },
    Category.COMPETITOR: {
        "zh": "抱歉，我無法提供其他品牌產品的規格、價格或比較，只能介紹 AORUS MASTER 16 AM6H。",
        "en": "Sorry, I can't provide specs, prices, or comparisons for other brands' products. I can only cover the AORUS MASTER 16 AM6H.",
    },
    Category.INJECTION: {
        "zh": "抱歉，我無法變更或略過系統規則。如果您有 AORUS MASTER 16 AM6H 的產品問題，歡迎直接詢問。",
        "en": "Sorry, I can't change or bypass my rules. If you have a question about the AORUS MASTER 16 AM6H, feel free to ask it directly.",
    },
    Category.NO_DATA: {
        "zh": "目前的產品資料中沒有這項資訊。",
        "en": "I don't have that information in the product data.",
    },
}


def refusal_message(category: Category, lang: str) -> str:
    """Fixed template per category; never echoes the user's text."""
    msg = REFUSALS.get(category, REFUSALS[Category.OFF_TOPIC])[lang]
    return f"{msg}{' ' if lang == 'en' else ''}{_SUGGEST[lang]}"


def refusal_text(lang: str) -> str:
    """Fallback when retrieval finds nothing relevant (no LLM call)."""
    return refusal_message(Category.NO_DATA, lang)


def detect_lang(text: str) -> str:
    """Return 'zh' if the text contains CJK characters, else 'en'.
    Mixed zh/en queries are answered in Chinese (target users are zh-TW)."""
    return "zh" if _CJK_RE.search(text) else "en"
