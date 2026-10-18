"""Text normalisation to resist evasion before any rule or lexicon matching.

Handles: full/half width and compatibility forms (NFKC), case, zero-width / invisible
characters, separators inserted inside words ("R.O.G", "r o g", "政 治", "政.治"),
Simplified Chinese variants of words in our lexicons, and a leetspeak variant
("a$us", "4lienware") used only for entity matching.
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE_RE = re.compile(r"[­​-‏‪-‮⁠-⁤﻿]")
_CJK = r"㐀-䶿一-鿿豈-﫿"
# separators users insert to break keyword matching
_SEP = r"[\s.\-_*·•/\\|~^'\"`、，,。]"
_CJK_SPLIT_RE = re.compile(rf"(?<=[{_CJK}]){_SEP}+(?=[{_CJK}])")
# three or more single latin letters/digits separated by separators: "r o g", "r.o.g"
_SPACED_LATIN_RE = re.compile(rf"(?<![a-z0-9])(?:[a-z0-9]{_SEP}+){{2,}}[a-z0-9](?![a-z0-9])")
_MULTISPACE_RE = re.compile(r"\s+")

# Simplified -> Traditional for characters that occur in brand / policy / spec vocabulary.
# A full converter (OpenCC) is unnecessary: we only need lexicon terms to match.
_S2T = str.maketrans(
    "华硕联尔夺灵国败苹笔电脑选举总统党两种视规则显价钱买卖对这个们为说么吗现开发设让机觉认该应独宁习龙龟师术号达护际红劳恋罗"
    "处忆体储键盘荐厂竞争无词输贴务员间页续热风评测试关门问题运动绝过还进远边后从来时会东张长马鸟鱼鲁黄孙陈刘杨赵谁赢数据图书"
    "学宫庙经兰罢众执胜军",
    "華碩聯爾奪靈國敗蘋筆電腦選舉總統黨兩種視規則顯價錢買賣對這個們為說麼嗎現開發設讓機覺認該應獨寧習龍龜師術號達護際紅勞戀羅"
    "處憶體儲鍵盤薦廠競爭無詞輸貼務員間頁續熱風評測試關門問題運動絕過還進遠邊後從來時會東張長馬鳥魚魯黃孫陳劉楊趙誰贏數據圖書"
    "學宮廟經蘭罷眾執勝軍",
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_LEET_TOKEN_RE = re.compile(r"[a-z0-9@$!]+")


def normalize(text: str) -> str:
    """Canonical form for rule matching (not shown to users or sent to the LLM)."""
    text = _INVISIBLE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text).translate(_S2T).lower()
    text = _CJK_SPLIT_RE.sub("", text)
    text = _SPACED_LATIN_RE.sub(lambda m: re.sub(_SEP, "", m.group(0)), text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def deleet(text: str) -> str:
    """Leetspeak variant for entity matching: only tokens mixing letters with digits/symbols
    are rewritten, so model numbers like "5090" or "275hx" stay mostly intact."""

    def fix(m: re.Match) -> str:
        tok = m.group(0)
        has_alpha = any(c.isalpha() for c in tok)
        has_sub = any(c in "0134 57@$!" for c in tok)
        return tok.translate(_LEET) if has_alpha and has_sub else tok

    return _LEET_TOKEN_RE.sub(fix, text)
