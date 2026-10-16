"""TTFT / TPS measurement around a streaming generator."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass
class GenStats:
    ttft_ms: float = 0.0  # request start -> first streamed text
    total_ms: float = 0.0
    n_tokens: int = 0  # generated tokens (re-tokenised output; stream events can merge tokens)
    text: str = ""

    @property
    def decode_tps(self) -> float:
        """Tokens/s after the first token (excludes prompt processing)."""
        gen_ms = self.total_ms - self.ttft_ms
        return (self.n_tokens - 1) / (gen_ms / 1000) if gen_ms > 0 and self.n_tokens > 1 else 0.0

    @property
    def e2e_tps(self) -> float:
        return self.n_tokens / (self.total_ms / 1000) if self.total_ms > 0 else 0.0


def measure_stream(
    stream: Iterator[str],
    stats: GenStats,
    count_tokens: Callable[[str], int] | None = None,
) -> Iterator[str]:
    """Pass-through wrapper: yields deltas unchanged and fills `stats` when exhausted.
    Start the clock *before* creating `stream` if prompt build time should count;
    here the clock starts on first iteration, which is when llama.cpp begins prefill."""
    t0 = time.perf_counter()
    parts: list[str] = []
    for delta in stream:
        if not parts:
            stats.ttft_ms = (time.perf_counter() - t0) * 1000
        parts.append(delta)
        yield delta
    stats.total_ms = (time.perf_counter() - t0) * 1000
    stats.text = "".join(parts)
    stats.n_tokens = count_tokens(stats.text) if count_tokens else len(parts)
