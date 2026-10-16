"""Fetch (optional) -> parse spec pages -> chunk.

Writes data/processed/specs.json and data/processed/chunks.jsonl.

    uv run python scripts/prepare_data.py [--fetch]
"""

import argparse
import json
from dataclasses import asdict

from aorus_assistant.config import DATA_DIR, PRODUCT_URLS, TARGET_PRODUCT, Config
from aorus_assistant.rag.chunker import chunk_specs
from aorus_assistant.rag.parser import parse_files
from aorus_assistant.rag.scraper import fetch_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="re-download HTML into data/raw/")
    args = parser.parse_args()

    raw = DATA_DIR / "raw"
    if args.fetch:
        fetch_all(raw)
    records = parse_files(
        raw / "spec_tw.html",
        raw / "spec_en.html",
        product=TARGET_PRODUCT,
        zh_url=PRODUCT_URLS["spec_tw"],
        en_url=PRODUCT_URLS["spec_en"],
    )
    cfg = Config()
    cfg.specs_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.specs_path.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    skus = sorted({r.sku for r in records})
    print(f"{len(records)} records, {len(skus)} SKUs {skus} -> {cfg.specs_path}")

    chunks = chunk_specs(records)
    with cfg.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks -> {cfg.chunks_path}")


if __name__ == "__main__":
    main()
