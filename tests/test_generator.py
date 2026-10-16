"""Prompt/context construction tests (no model needed)."""

from aorus_assistant.llm.generator import build_messages, format_context
from aorus_assistant.schemas import Chunk, RetrievedChunk


def _rc(cid, section, skus, lines, field_zh="欄位"):
    meta = {"field_zh": field_zh, "field_en": section, "skus": skus, "lines": lines, "notes": []}
    return RetrievedChunk(Chunk(cid, "", "AORUS MASTER 16 AM6H", section, "u", meta), score=1.0)


def test_per_sku_chunks_are_pivoted_in_flagship_order():
    ctx = [
        _rc("g-x", "Video Graphics", ["AORUS MASTER 16 BXH"], ["NVIDIA GeForce RTX 5070 Ti Laptop GPU", "12GB GDDR7"]),
        _rc("g-z", "Video Graphics", ["AORUS MASTER 16 BZH"], ["NVIDIA GeForce RTX 5090 Laptop GPU", "24GB GDDR7"]),
    ]
    text = format_context(ctx)
    assert "依版本不同" in text
    assert "顯示記憶體 VRAM: BZH 版 24GB GDDR7 / BXH 版 12GB GDDR7" in text


def test_lineup_rendered_last_and_shared_field_labelled():
    ctx = [
        RetrievedChunk(
            Chunk("lineup", "", "p", "SKU Lineup", "u",
                  {"field_zh": "版本總覽", "field_en": "SKU Lineup", "skus": [], "lines": ["L"], "notes": []}),
            score=1.0,
        ),
        _rc("d", "Display", ["A", "B", "C"], ["OLED WQXGA (2560×1600) 240Hz, 1ms"], "顯示器"),
    ]
    text = format_context(ctx)
    assert text.index("顯示器") < text.index("版本總覽")
    assert "全系列相同" in text and "更新率 Refresh rate: 240Hz" in text


def test_language_instruction_follows_query():
    zh = build_messages("電池多大", [], "zh")[1]["content"]
    en = build_messages("battery?", [], "en")[1]["content"]
    assert zh.rstrip().endswith("請用繁體中文（台灣用語）回答。")
    assert en.rstrip().endswith("Answer in English.")
