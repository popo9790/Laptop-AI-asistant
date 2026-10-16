"""Tokenizer / BM25 / hybrid retriever tests with a fake embedder (no model needed)."""

import numpy as np

from aorus_assistant.config import RetrievalConfig
from aorus_assistant.rag.index import BM25, VectorIndex
from aorus_assistant.rag.retriever import Retriever
from aorus_assistant.rag.tokenize import tokenize
from aorus_assistant.schemas import Chunk


def test_tokenize_mixed():
    toks = tokenize("AM6H 的顯卡是 RTX™ 5070 Ti 嗎？")
    assert {"am6h", "rtx", "5070", "ti"} <= set(toks)
    assert "顯卡" in toks and "顯" in toks  # CJK bigram + unigram
    assert tokenize("ＤＤＲ５") == ["ddr5"]  # full-width normalised
    assert "wi-fi" in tokenize("Wi-Fi 7")


def test_bm25_ranks_exact_term_first():
    docs = [tokenize(t) for t in ["電池 99Wh battery", "螢幕 OLED 240Hz", "重量 2.5 kg"]]
    s = BM25(docs).scores(tokenize("電池容量多少"))
    assert int(np.argmax(s)) == 0
    assert BM25(docs).scores(tokenize("完全無關 xyz")).max() == 0


def _chunk(i, text, product="AORUS MASTER 16 AM6H"):
    return Chunk(id=f"c{i}", text=text, product=product, section="s", source_url="u")


def test_scope_filter_excludes_before_scoring():
    chunks = [_chunk(0, "battery 99Wh"), _chunk(1, "battery 90Wh", product="OTHER LAPTOP")]
    vecs = np.eye(2, dtype=np.float32)
    retr = Retriever(
        chunks,
        VectorIndex(vecs, ["c0", "c1"]),
        embed_query=lambda qs: np.array([[0.0, 1.0]], dtype=np.float32),  # dense favours c1
        cfg=RetrievalConfig(top_k=5),
        chunk_filter=lambda c: c.product == "AORUS MASTER 16 AM6H",
    )
    ids = [r.chunk.id for r in retr.retrieve("battery")]
    assert ids == ["c0"]


def test_index_roundtrip(tmp_path):
    idx = VectorIndex(np.random.rand(3, 4).astype(np.float32), ["a", "b", "c"], "m.gguf")
    idx.save(tmp_path / "i.npz")
    loaded = VectorIndex.load(tmp_path / "i.npz")
    assert loaded.ids == ["a", "b", "c"] and loaded.model_name == "m.gguf"
    np.testing.assert_allclose(loaded.embeddings, idx.embeddings)


def test_stopwords_removed():
    assert tokenize("What is the battery capacity?") == ["battery", "capacity"]
    toks = tokenize("這台多重？")
    assert "這台" not in toks and "重" in toks
