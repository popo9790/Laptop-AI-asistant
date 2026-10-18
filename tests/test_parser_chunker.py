"""Parser / chunker tests against the cached spec pages in data/raw/ (no network)."""

import pytest

from aorus_assistant.config import DATA_DIR, PRODUCT_URLS, TARGET_PRODUCT
from aorus_assistant.rag.chunker import chunk_specs
from aorus_assistant.rag.parser import parse_files

RAW = DATA_DIR / "raw"
pytestmark = pytest.mark.skipif(
    not (RAW / "spec_tw.html").exists(), reason="cached HTML missing; run prepare_data.py --fetch"
)


@pytest.fixture(scope="module")
def records():
    return parse_files(
        RAW / "spec_tw.html", RAW / "spec_en.html", TARGET_PRODUCT,
        PRODUCT_URLS["spec_tw"], PRODUCT_URLS["spec_en"],
    )


def _get(records, sku_suffix, field_en):
    return next(r for r in records if r.sku.endswith(sku_suffix) and r.field_en == field_en)


def test_skus_and_fields(records):
    assert sorted({r.sku for r in records}) == [
        "AORUS MASTER 16 BXH", "AORUS MASTER 16 BYH", "AORUS MASTER 16 BZH",
    ]
    assert len(records) == 3 * 17
    assert all(r.lines for r in records), "every field should have a value"


def test_sku_gpu_mapping(records):
    # verified against the rendered comparison table on gigabyte.com/tw
    assert "RTX™ 5090" in _get(records, "BZH", "Video Graphics").lines[0]
    assert "RTX™ 5080" in _get(records, "BYH", "Video Graphics").lines[0]
    assert "RTX™ 5070 Ti" in _get(records, "BXH", "Video Graphics").lines[0]


def test_key_values(records):
    assert _get(records, "BZH", "CPU").field_zh == "中央處理器"
    assert _get(records, "BZH", "Battery").lines == ["Li-ion 99Wh"]
    assert _get(records, "BZH", "System Memory").lines == ["Up to 64GB DDR5 5600MHz"]
    gpu = _get(records, "BZH", "Video Graphics")
    assert gpu.notes == ["*May vary by scenario"]  # footnotes split from value lines


def test_chunks_dedupe_shared_fields(records):
    chunks = chunk_specs(records)
    gpu_chunks = [c for c in chunks if c.section == "Video Graphics"]
    shared = [c for c in chunks if c.id.endswith("-all")]
    assert len(gpu_chunks) == 3
    assert len(shared) == 16
    assert len(chunks) == 16 + 3 + 2  # shared + per-SKU GPU + lineup + performance overview
    assert all(c.product == TARGET_PRODUCT for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_text_bilingual(records):
    chunks = {c.id: c for c in chunk_specs(records)}
    display = next(c for c in chunks.values() if c.section == "Display")
    assert "顯示器" in display.text and "Display" in display.text and "螢幕" in display.text
    lineup = chunks["lineup"]
    for s in ("BZH", "BYH", "BXH", "5090", "5080", "5070 Ti"):
        assert s in lineup.text
    perf = chunks["perf"]
    for s in ("275HX", "5090", "5070 Ti", "64GB", "剪輯", "rendering"):
        assert s in perf.text
