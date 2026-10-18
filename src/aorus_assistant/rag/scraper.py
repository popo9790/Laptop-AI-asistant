"""Fetch raw HTML of the product pages and cache it under data/raw/.

gigabyte.com sits behind Akamai, which returns 403 for bare clients: it needs a full set of
browser navigation headers *and* an HTTP/2 client (`requests`, HTTP/1.1 only, is rejected
even with the same headers), hence httpx with http2.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from aorus_assistant.config import DATA_DIR, PRODUCT_URLS

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}


def fetch(url: str, out_path: Path, timeout: float = 30) -> Path:
    with httpx.Client(http2=True, headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resp.text, encoding="utf-8")
    return out_path


def fetch_all(out_dir: Path = DATA_DIR / "raw") -> dict[str, Path]:
    return {name: fetch(url, out_dir / f"{name}.html") for name, url in PRODUCT_URLS.items()}
