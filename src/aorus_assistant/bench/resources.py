"""Memory measurement: llama.cpp buffer accounting, NVML VRAM, process RSS, peak sampling.

Two complementary views:
  - llama.cpp buffer accounting (parsed from its load log): exactly what each model
    allocates per device (weights / KV cache / recurrent state / compute). Platform
    independent, so a Mac (unified memory, no VRAM counter) still yields the VRAM budget
    the same config needs on a discrete GPU.
  - Live counters sampled during a request: NVML device/process VRAM on NVIDIA, and
    process RSS everywhere. These confirm the accounting and catch lazy allocations.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil

_BUF_RE = re.compile(
    r"(?P<dev>[A-Za-z0-9_]+?)(?:_Mapped|_REPACK)?\s+(?P<kind>model|KV|RS|compute|output) buffer size\s*=\s*"
    r"(?P<mib>[\d.]+) MiB"
)
CPU_DEVICES = {"CPU"}


@contextmanager
def capture_stderr_fd():
    """Capture C-level stderr (llama.cpp / ggml log) into a string: yields a list that
    receives the captured text on exit."""
    out: list[str] = []
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield out
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            out.append(tmp.read().decode("utf-8", errors="replace"))


def parse_llama_buffers(log: str) -> dict[str, dict[str, float]]:
    """{device: {kind: MiB}} from a verbose llama.cpp load log. Device names: CPU, MTL0
    (Apple Metal), CUDA0, ... Mapped/repacked weights are folded into `model`."""
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for line in log.splitlines():
        if line.startswith("~llama_context"):  # teardown echo of the same buffers
            continue
        m = _BUF_RE.search(line)
        if m:
            acc[m["dev"]][m["kind"]] += float(m["mib"])
    return {dev: dict(kinds) for dev, kinds in acc.items()}


def split_by_device(buffers: dict[str, dict[str, float]]) -> tuple[float, float]:
    """(accelerator MiB, CPU MiB)."""
    gpu = sum(sum(k.values()) for d, k in buffers.items() if d not in CPU_DEVICES)
    cpu = sum(sum(k.values()) for d, k in buffers.items() if d in CPU_DEVICES)
    return gpu, cpu


# ---------------------------------------------------------------- live counters
_nvml = None


def _nvml_handle():
    global _nvml
    if _nvml is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:  # no NVIDIA driver / GPU
            _nvml = False
    return _nvml or None


def vram_used_mb() -> float | None:
    """Device-wide used VRAM in MB (includes CUDA context), or None without NVIDIA GPU."""
    h = _nvml_handle()
    if not h:
        return None
    pynvml, dev = h
    return pynvml.nvmlDeviceGetMemoryInfo(dev).used / 2**20


def vram_process_mb(pid: int | None = None) -> float | None:
    h = _nvml_handle()
    if not h:
        return None
    pynvml, dev = h
    pid = pid or os.getpid()
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(dev)
    except Exception:
        return None
    used = [p.usedGpuMemory for p in procs if p.pid == pid and p.usedGpuMemory]
    return sum(used) / 2**20 if used else None


def ram_used_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


@dataclass
class PeakSampler:
    """Background sampler for peak RSS / VRAM while a block runs."""

    interval_s: float = 0.02
    peak_rss_mb: float = 0.0
    peak_vram_mb: float | None = None
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval_s)

    def _sample(self) -> None:
        self.peak_rss_mb = max(self.peak_rss_mb, ram_used_mb())
        v = vram_used_mb()
        if v is not None:
            self.peak_vram_mb = max(self.peak_vram_mb or 0.0, v)
        self.samples += 1

    def __enter__(self) -> "PeakSampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
