"""Stage 5 - Audit & explainability: append one JSON line per request."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from aorus_assistant.schemas import ModerationDecision


class AuditLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, request_id: str, query: str, decisions: list[ModerationDecision], **extra) -> None:
        record = {
            "ts": time.time(),
            "request_id": request_id,
            "query": query,
            "decisions": [asdict(d) for d in decisions],
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
