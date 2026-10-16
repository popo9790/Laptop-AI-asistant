"""VRAM (NVML) and RAM (psutil) sampling."""

from __future__ import annotations


def vram_used_mb() -> float | None:
    """Current process GPU memory in MB, or None when no NVIDIA GPU is present."""
    raise NotImplementedError


def ram_used_mb() -> float:
    raise NotImplementedError
