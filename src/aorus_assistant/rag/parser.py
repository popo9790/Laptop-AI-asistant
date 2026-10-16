"""Parse cached spec-page HTML into structured key-value SpecRecords.

Page structure (server-rendered by Nuxt):
  .multiple-spec-content-wrapper
    .spec-column > .multiple-title            field names, one per row
    .content-column .swiper-slide             one slide per SKU
      .spec-item-list[data-spec-row=N] > span field value, lines split by <br>
SKU names are not in the slide markup (filled client-side); they come from the Nuxt
payload's `productName` list, which has the same order as the slides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from aorus_assistant.schemas import SpecRecord

SKU_RE = re.compile(r'"productKey":\d+,"productName":(\d+)')


@dataclass
class ParsedPage:
    fields: list[str]
    skus: list[str]
    values: list[list[tuple[list[str], list[str]]]]  # [sku][row] -> (lines, notes)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_value(span: Tag) -> tuple[list[str], list[str]]:
    """Split a value cell into content lines and footnotes (<p> blocks / lines starting with '*')."""
    notes: list[str] = []
    for p in span.find_all("p"):
        notes.extend(_clean(n) for n in p.get_text("\n").split("\n") if _clean(n))
        p.decompose()
    lines = [_clean(x) for x in span.get_text("\n").split("\n")]
    lines = [x for x in lines if x]
    return lines, notes


def _sku_names(html: str, n_slides: int) -> list[str]:
    """Resolve SKU names from the Nuxt payload (a flat array with index references)."""
    names: list[str] = []
    payload_names = re.findall(r'"modelName":(\d+)[^}]*\},"\d+","(AORUS[^"]+)"', html)
    by_ref = {ref: name for ref, name in payload_names}
    for ref in SKU_RE.findall(html):
        if ref in by_ref and by_ref[ref] not in names:
            names.append(by_ref[ref])
    if len(names) != n_slides:  # fall back to positional placeholders rather than guessing
        names = [f"SKU-{i + 1}" for i in range(n_slides)]
    return names


def parse_spec_html(html: str) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    wrapper = soup.select_one(".multiple-spec-content-wrapper")
    if wrapper is None:
        raise ValueError("spec table (.multiple-spec-content-wrapper) not found")

    fields = [_clean(t.get_text(" ")) for t in wrapper.select(".spec-column .multiple-title")]
    slides = wrapper.select(".content-column .swiper-slide")
    values = []
    for slide in slides:
        rows: list[tuple[list[str], list[str]]] = [([], [])] * len(fields)
        for cell in slide.select(".spec-item-list[data-spec-row]"):
            span = cell.select_one(".swiper-no-swiping") or cell
            rows[int(cell["data-spec-row"])] = _split_value(span)
        values.append(rows)
    return ParsedPage(fields=fields, skus=_sku_names(html, len(slides)), values=values)


def build_records(
    zh: ParsedPage, en: ParsedPage, product: str, zh_url: str, en_url: str
) -> list[SpecRecord]:
    """Merge zh-TW and English pages by (sku, row). Values come from the zh-TW page
    (Taiwan-market SKUs); English supplies field names for bilingual retrieval."""
    if len(zh.fields) != len(en.fields):
        raise ValueError(f"field count mismatch: zh={len(zh.fields)} en={len(en.fields)}")
    records = []
    for s, sku in enumerate(zh.skus):
        for r, field_zh in enumerate(zh.fields):
            lines, notes = zh.values[s][r]
            records.append(
                SpecRecord(
                    product=product,
                    sku=sku,
                    row=r,
                    field_zh=field_zh,
                    field_en=en.fields[r],
                    lines=lines,
                    notes=notes,
                    source_urls=[zh_url, en_url],
                )
            )
    return records


def parse_files(zh_path: Path, en_path: Path, product: str, zh_url: str, en_url: str) -> list[SpecRecord]:
    zh = parse_spec_html(zh_path.read_text(encoding="utf-8"))
    en = parse_spec_html(en_path.read_text(encoding="utf-8"))
    return build_records(zh, en, product, zh_url, en_url)
