"""TTFT / TPS measurement around a streaming generator."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class GenStats:
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    n_tokens: int = 0

    @property
    def tps(self) -> float:
        gen_ms = self.total_ms - self.ttft_ms
        return self.n_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0


def measure_stream(stream: Iterator[str], stats: GenStats) -> Iterator[str]:
    raise NotImplementedError
