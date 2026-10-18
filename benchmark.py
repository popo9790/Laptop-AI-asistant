"""Performance benchmark: moderation off vs on, same queries, same models.

    uv run python benchmark.py                 # 3 runs over the benchmark queries
    uv run python benchmark.py --runs 1 --llm-file Qwen3.5-0.8B-Q4_K_M.gguf

Measures per request: TTFT (request start -> first text shown to the user, so it includes the
query embedding, moderation and the output guard's hold-back), LLM TTFT (prefill only),
decode TPS (LLM tokens after the first), end-to-end latency,
and per-stage moderation overhead. Memory: llama.cpp buffer accounting and process peaks
(NVML VRAM on NVIDIA). Queries are allowed product questions so both modes run the LLM;
refusals would make the moderated mode look artificially fast.
Writes logs/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from aorus_assistant.bench.resources import (
    PeakSampler,
    capture_stderr_fd,
    parse_llama_buffers,
    ram_used_mb,
    split_by_device,
    vram_used_mb,
)
from aorus_assistant.config import Config
from aorus_assistant.llm.models import load_llm
from aorus_assistant.pipeline import AssistantPipeline
from aorus_assistant.rag.embedder import Embedder

QUERIES = [
    "這台筆電的顯示卡是什麼？",
    "記憶體最多可以擴充到多少？",
    "What is the battery capacity?",
    "有支援 Thunderbolt 5 嗎？",
    "螢幕的 refresh rate 是多少",
    "這台有哪幾個版本？差在哪裡",
    "How much does it weigh?",
    "CPU 幾核心？",
    "可以裝兩條 SSD 嗎",
    "HDMI 是哪個版本",
]


def stats(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return {"mean": round(float(np.mean(xs)), 1), "p50": round(float(np.percentile(xs, 50)), 1),
            "p95": round(float(np.percentile(xs, 95)), 1)}


def run_interleaved(pipes: dict[str, AssistantPipeline], runs: int) -> dict[str, list[dict]]:
    """Alternate modes query by query so thermal / background drift on a laptop affects both
    equally (running one mode after the other biased the second)."""
    rows: dict[str, list[dict]] = {name: [] for name in pipes}
    for pipe in pipes.values():
        pipe.answer(QUERIES[0])  # warm-up (Metal/CUDA kernel compilation)
    for _ in range(runs):
        for q in QUERIES:
            for name, pipe in pipes.items():
                r = pipe.answer(q)
                rows[name].append({
                    "query": q, "refused": r.refused, "refusal_stage": r.refusal_stage,
                    "redacted": r.redacted, "n_tokens": r.n_tokens,
                    **{k: round(v, 2) for k, v in r.timings_ms.items()}, "decode_tps": r.decode_tps,
                })
    return rows


def summarize(rows: list[dict]) -> dict:
    moderation_ms = [
        r.get("input_guard", 0) + r.get("retrieval_filter", 0) + r.get("output_guard", 0) for r in rows
    ]
    return {
        "n_requests": len(rows),
        "refused": sum(r["refused"] for r in rows),
        "ttft_ms": stats(r.get("ttft") for r in rows),
        "llm_ttft_ms": stats(r.get("llm_ttft") for r in rows),
        "total_ms": stats(r.get("total") for r in rows),
        "decode_tps": stats(r["decode_tps"] for r in rows),
        "embed_ms": stats(r.get("embed") for r in rows),
        "retrieval_ms": stats(r.get("retrieval") for r in rows),
        "input_guard_ms": stats(r.get("input_guard") for r in rows),
        "retrieval_filter_ms": stats(r.get("retrieval_filter") for r in rows),
        "output_guard_ms": stats(r.get("output_guard") for r in rows),
        "moderation_compute_ms": stats(moderation_ms),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--llm-file")
    ap.add_argument("--out", default="logs/benchmark.json")
    args = ap.parse_args()

    cfg = Config()
    if args.llm_file:
        cfg.llm.filename = args.llm_file
    vram0 = vram_used_mb()
    with capture_stderr_fd() as emb_log:
        embedder = Embedder(cfg.embed, verbose=True)
    with capture_stderr_fd() as llm_log:
        llm = load_llm(cfg.llm, verbose=True)
    llm.verbose = embedder.model.verbose = False
    llm_gpu, llm_cpu = split_by_device(parse_llama_buffers(llm_log[0]))
    emb_gpu, emb_cpu = split_by_device(parse_llama_buffers(emb_log[0]))

    report = {
        "platform": f"{platform.system()} {platform.machine()}",
        "llm": cfg.llm.filename, "embed": cfg.embed.filename,
        "n_ctx": cfg.llm.n_ctx, "n_ubatch": cfg.llm.n_ubatch,
        "memory": {
            "accelerator_buffers_mib": round(llm_gpu + emb_gpu, 1),
            "cpu_buffers_mib": round(llm_cpu + emb_cpu, 1),
            "vram_before_load_mb": vram0,
            "vram_after_load_mb": vram_used_mb(),
            "rss_after_load_mb": round(ram_used_mb(), 1),
        },
    }
    pipes = {name: AssistantPipeline(cfg, moderation=mod, llm=llm, embedder=embedder, audit=False)
             for name, mod in (("moderation_off", False), ("moderation_on", True))}
    with PeakSampler() as peak:
        rows = run_interleaved(pipes, args.runs)
    for name in pipes:
        report[name] = summarize(rows[name])
    report["memory"]["peak_rss_mb"] = round(peak.peak_rss_mb, 1)
    report["memory"]["peak_vram_mb"] = round(peak.peak_vram_mb, 1) if peak.peak_vram_mb else None

    off, on = report["moderation_off"], report["moderation_on"]
    print(f"platform {report['platform']}  LLM {cfg.llm.filename}  runs={args.runs} x {len(QUERIES)} queries")
    print(f"memory: accelerator buffers {report['memory']['accelerator_buffers_mib']} MiB, "
          f"CPU buffers {report['memory']['cpu_buffers_mib']} MiB, VRAM after load {report['memory']['vram_after_load_mb']}")
    print(f"{'metric':<26}{'moderation off':>18}{'moderation on':>18}")
    for key in ("ttft_ms", "llm_ttft_ms", "total_ms", "decode_tps", "embed_ms", "retrieval_ms"):
        for s in ("p50", "mean"):
            print(f"{key + ' ' + s:<26}{off[key].get(s, '-'):>18}{on[key].get(s, '-'):>18}")
    for key in ("input_guard_ms", "retrieval_filter_ms", "output_guard_ms", "moderation_compute_ms"):
        print(f"{key + ' p50':<26}{'-':>18}{on[key].get('p50', '-'):>18}")
    print(f"{'refused (should be 0)':<26}{off['refused']:>18}{on['refused']:>18}")
    print(f"peak RSS {report['memory']['peak_rss_mb']} MB, peak VRAM (NVML) {report['memory']['peak_vram_mb']} "
          "(both modes share one process and the same loaded models)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
