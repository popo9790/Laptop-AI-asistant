"""Spec key-value data -> retrieval chunks.

    uv run python scripts/prepare_data.py            # data/specs.json -> data/processed/chunks.jsonl
    uv run python scripts/prepare_data.py --fetch    # re-scrape the product pages first:
                                                     # HTML -> data/raw/ (not committed) -> parse -> data/specs.json
"""

import argparse
import json
from dataclasses import asdict

from aorus_assistant.config import DATA_DIR, PRODUCT_URLS, TARGET_PRODUCT, Config
from aorus_assistant.rag.chunker import chunk_specs
from aorus_assistant.schemas import SpecRecord


def fetch_and_parse(cfg: Config) -> None:
    from aorus_assistant.rag.parser import parse_files
    from aorus_assistant.rag.scraper import fetch_all

    raw = DATA_DIR / "raw"
    fetch_all(raw)
    records = parse_files(
        raw / "spec_tw.html",
        raw / "spec_en.html",
        product=TARGET_PRODUCT,
        zh_url=PRODUCT_URLS["spec_tw"],
        en_url=PRODUCT_URLS["spec_en"],
    )
    cfg.specs_path.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"parsed {len(records)} records -> {cfg.specs_path}")


def load_specs(cfg: Config) -> list[SpecRecord]:
    return [SpecRecord(**d) for d in json.loads(cfg.specs_path.read_text(encoding="utf-8"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="re-scrape and re-parse the product pages")
    args = parser.parse_args()

    cfg = Config()
    if args.fetch:
        fetch_and_parse(cfg)
    records = load_specs(cfg)
    skus = sorted({r.sku for r in records})
    print(f"{len(records)} records, {len(skus)} SKUs {skus} <- {cfg.specs_path}")

    chunks = chunk_specs(records)
    cfg.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks -> {cfg.chunks_path}")


if __name__ == "__main__":
    main()
