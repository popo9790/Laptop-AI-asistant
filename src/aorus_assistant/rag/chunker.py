"""Turn SpecRecords into retrieval chunks.

Strategy (the spec table is small and highly structured, so chunk by field, not by length):
  - fields identical across SKUs -> one shared chunk (avoids 3 near-duplicate chunks crowding top-k)
  - fields that differ           -> one chunk per SKU
  - one SKU-lineup chunk summarising the differing fields, for "which versions exist" questions
  - one performance-overview chunk (CPU + GPU per SKU + memory + storage) for workload
    questions ("剪 4K 影片會不會卡", "can it run a local LLM") that no single field answers
Each chunk carries zh + en field names plus colloquial aliases so BM25 can match
zh-TW, English, and mixed queries.
"""

from __future__ import annotations

from collections import defaultdict

from aorus_assistant.schemas import Chunk, SpecRecord

# Colloquial terms users actually type, keyed by English field name from the spec page.
FIELD_ALIASES: dict[str, list[str]] = {
    "OS": ["作業系統", "系統", "Windows", "operating system"],
    "CPU": ["處理器", "中央處理器", "processor", "核心", "時脈"],
    "Video Graphics": ["顯示卡", "顯卡", "獨顯", "GPU", "graphics card", "VRAM", "顯存", "TGP"],
    "Display": ["螢幕", "顯示器", "面板", "screen", "panel", "更新率", "refresh rate", "解析度", "resolution"],
    "System Memory": ["記憶體", "RAM", "內存", "memory", "擴充", "upgrade"],
    "Storage": ["儲存", "硬碟", "SSD", "固態硬碟", "容量", "M.2", "slot"],
    "Keyboard Type": ["鍵盤", "背光", "RGB", "keyboard", "鍵程"],
    "I/O Port": ["連接埠", "接口", "插孔", "ports", "USB", "HDMI", "Thunderbolt", "Type-C", "讀卡機"],
    "Audio": ["音效", "喇叭", "揚聲器", "speakers", "麥克風"],
    "Communications": ["網路", "無線", "WiFi", "Wi-Fi", "藍牙", "Bluetooth", "LAN", "網卡"],
    "Webcam": ["視訊鏡頭", "鏡頭", "攝影機", "camera", "人臉辨識", "Windows Hello"],
    "Security": ["安全", "TPM", "security"],
    "Battery": ["電池", "電量", "續航", "battery", "Wh"],
    "Adapter": ["變壓器", "充電器", "電源", "adapter", "charger", "power supply", "瓦數"],
    "Dimensions (W x D x H)": ["尺寸", "大小", "厚度", "長寬高", "size", "dimensions", "thickness"],
    "Weight": ["重量", "多重", "幾公斤", "weight", "heavy"],
    "Color": ["顏色", "外觀", "color", "colour"],
}


def _short_sku(sku: str) -> str:
    return sku.split()[-1]  # "AORUS MASTER 16 BZH" -> "BZH"


def _field_header(rec: SpecRecord) -> str:
    return f"{rec.field_zh} / {rec.field_en}"


def _aliases_line(rec: SpecRecord) -> str:
    aliases = FIELD_ALIASES.get(rec.field_en, [])
    return f"關鍵字 Keywords: {', '.join(aliases)}" if aliases else ""


def _render(rec: SpecRecord, sku_label: str) -> str:
    parts = [
        f"{rec.product}（{sku_label}）",
        f"規格項目 Spec: {_field_header(rec)}",
        *rec.lines,
    ]
    notes = [n for n in rec.notes if not n.startswith("http")]
    if notes:
        parts.append("註 Notes: " + " ".join(notes))
    if alias := _aliases_line(rec):
        parts.append(alias)
    return "\n".join(parts)


def _make_chunk(rec: SpecRecord, skus: list[str], chunk_id: str) -> Chunk:
    label = "全系列 All SKUs: " + " / ".join(skus) if len(skus) > 1 else f"SKU: {skus[0]}"
    return Chunk(
        id=chunk_id,
        text=_render(rec, label),
        product=rec.product,
        section=rec.field_en,
        source_url=rec.source_urls[0],
        metadata={
            "field_zh": rec.field_zh,
            "field_en": rec.field_en,
            "row": rec.row,
            "skus": skus,
            "source_urls": rec.source_urls,
            # Clean content for the LLM prompt (chunk.text also carries retrieval-only
            # aliases and long SKU headers that distract a small model).
            "lines": rec.lines,
            "notes": [n for n in rec.notes if not n.startswith("http")],
        },
    )


def _lineup_chunk(product: str, diff_rows: list[list[SpecRecord]], source_url: str) -> Chunk:
    skus = [r.sku for r in diff_rows[0]] if diff_rows else []
    lines = [
        f"{product} 共有 {len(skus)} 種配置 / {len(skus)} configurations (SKU):",
    ]
    for i, sku in enumerate(skus):
        diffs = "; ".join(f"{row[i].field_zh}: {', '.join(row[i].lines)}" for row in diff_rows)
        lines.append(f"- {sku} ({_short_sku(sku)}): {diffs}")
    lines.append("其餘規格各版本相同 / All other specs are identical across SKUs.")
    # Avoid generic words like "版本" here: "HDMI 版本" / "藍牙版本" would all hit this chunk.
    lines.append("關鍵字 Keywords: 型號, 配置, 規格差異, SKU, configurations, models, variants")
    return Chunk(
        id="lineup",
        text="\n".join(lines),
        product=product,
        section="SKU Lineup",
        source_url=source_url,
        metadata={
            "skus": skus,
            "field_en": "SKU Lineup",
            "field_zh": "版本總覽",
            "lines": lines[:-1],  # without the keyword line
            "notes": [],
        },
    )


PERF_FIELDS = ["CPU", "Video Graphics", "System Memory", "Storage"]
WORKLOAD_KEYWORDS = (
    "效能, 剪輯, 剪片, 影片剪輯, 4K, 8K, 渲染, 3D, 建模, 遊戲, 3A, 電競, 直播, 推流, 實況, 調色, 後製, "
    "AI, 深度學習, 訓練模型, 本地 LLM, 多工, 夠不夠力, 會不會卡, performance, workload, video editing, "
    "rendering, gaming, streaming, color grading, machine learning, local LLM, Premiere, DaVinci, Blender"
)


def _perf_chunk(by_row: dict[int, list[SpecRecord]]) -> Chunk | None:
    rows = {recs[0].field_en: recs for recs in by_row.values()}
    if not all(f in rows for f in PERF_FIELDS):
        return None
    product = rows["CPU"][0].product
    lines = [f"{product} 效能相關規格 / Performance-related specs:"]
    for field_en in PERF_FIELDS:
        recs = rows[field_en]
        if len({tuple(r.lines) for r in recs}) == 1:
            lines.append(f"{recs[0].field_zh} {field_en}（全系列 All SKUs）: {', '.join(recs[0].lines)}")
        else:
            per = " / ".join(f"{_short_sku(r.sku)} 版: {', '.join(r.lines)}" for r in recs)
            lines.append(f"{recs[0].field_zh} {field_en}（依版本不同）: {per}")
    return Chunk(
        id="perf",
        text="\n".join(lines + [f"關鍵字 Keywords: {WORKLOAD_KEYWORDS}"]),
        product=product,
        section="Performance Overview",
        source_url=rows["CPU"][0].source_urls[0],
        metadata={
            "field_zh": "效能總覽",
            "field_en": "Performance Overview",
            "skus": [r.sku for r in rows["Video Graphics"]],
            "lines": lines,
            "notes": [],
        },
    )


def chunk_specs(records: list[SpecRecord]) -> list[Chunk]:
    by_row: dict[int, list[SpecRecord]] = defaultdict(list)
    for rec in records:
        by_row[rec.row].append(rec)

    chunks: list[Chunk] = []
    diff_rows: list[list[SpecRecord]] = []
    for row in sorted(by_row):
        recs = by_row[row]
        if len({tuple(r.lines) for r in recs}) == 1:
            chunks.append(_make_chunk(recs[0], [r.sku for r in recs], f"row{row:02d}-all"))
        else:
            diff_rows.append(recs)
            for r in recs:
                chunks.append(_make_chunk(r, [r.sku], f"row{row:02d}-{_short_sku(r.sku)}"))

    if records:
        chunks.append(_lineup_chunk(records[0].product, diff_rows, records[0].source_urls[0]))
        if perf := _perf_chunk(by_row):
            chunks.append(perf)
    return chunks
