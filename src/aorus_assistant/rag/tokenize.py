"""Lightweight tokenizer for BM25 that handles zh-TW / English / mixed text.

No jieba dependency: CJK runs are split into character unigrams + bigrams, which is a
standard, dictionary-free baseline for Chinese IR. Latin/number runs are lower-cased
words; tokens like "5070", "DDR5", "Wi-Fi" are kept intact.
"""

from __future__ import annotations

import re
import unicodedata

_CJK = r"㐀-䶿一-鿿豈-﫿"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STRIP_MARKS = str.maketrans("", "", "®™©")

# Function words that carry no spec meaning; without these, English questions like
# "What is the battery capacity?" match footnotes ("The storage capacity is ...").
STOPWORDS = frozenset(
    "a an the is are was be do does did it its this that these those of to in on for with "
    "and or what which how much many can could i my me you your there have has any about "
    "laptop notebook".split()
) | frozenset("的 是 嗎 呢 吧 啊 有 我 你 這 那 台 個 了 和 跟 與 請 問 多 少 幾 什 麼 哪".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text.translate(_STRIP_MARKS)).lower()
    tokens: list[str] = []
    for run in _TOKEN_RE.findall(text):
        if re.match(rf"[{_CJK}]", run):
            tokens.extend(ch for ch in run if ch not in STOPWORDS)  # unigrams
            # bigrams made only of stopwords ("這台", "多少") are dropped too
            tokens.extend(
                run[i : i + 2]
                for i in range(len(run) - 1)
                if not (run[i] in STOPWORDS and run[i + 1] in STOPWORDS)
            )
        elif run not in STOPWORDS:
            tokens.append(run)
    return tokens
