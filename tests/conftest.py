import json
from pathlib import Path

import pytest

CASES_DIR = Path(__file__).parent / "cases"


def load_cases(name: str) -> list[dict]:
    path = CASES_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def all_cases() -> list[dict]:
    return [
        c for p in sorted(CASES_DIR.glob("*.jsonl")) if p.stem not in ("retrieval", "rag_qa") for c in load_cases(p.stem)
    ]
