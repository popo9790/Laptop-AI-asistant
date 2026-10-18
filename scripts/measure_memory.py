"""Measure memory for one full answer: embedder + LLM load, retrieval, streaming generation.

    uv run python scripts/measure_memory.py
    uv run python scripts/measure_memory.py --llm-file Qwen3.5-0.8B-Q4_K_M.gguf --n-ubatch 128

Reports llama.cpp buffer accounting per device (weights / KV / recurrent state / compute),
live process RSS and, on NVIDIA GPUs, NVML VRAM (device-wide and per-process), including
peaks sampled every 20 ms while answering. Writes logs/memory.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aorus_assistant.bench.resources import (  # noqa: E402
    PeakSampler,
    capture_stderr_fd,
    parse_llama_buffers,
    ram_used_mb,
    split_by_device,
    vram_process_mb,
    vram_used_mb,
)
from aorus_assistant.bench.timing import GenStats, measure_stream  # noqa: E402
from aorus_assistant.config import Config  # noqa: E402
from aorus_assistant.llm.generator import build_messages, stream_answer  # noqa: E402
from aorus_assistant.llm.models import load_llm  # noqa: E402
from aorus_assistant.moderation.responses import detect_lang  # noqa: E402
from aorus_assistant.rag.embedder import Embedder  # noqa: E402
from aorus_assistant.rag.index import VectorIndex  # noqa: E402
from aorus_assistant.rag.retriever import Retriever, load_chunks  # noqa: E402

# Largest prompt in the eval set (3 per-SKU GPU chunks + lineup) and a long answer.
DEFAULT_QUERIES = ["這台有哪幾個版本？差在哪裡", "這台筆電的顯示卡是什麼？", "What is the battery capacity?"]


def snapshot(label: str) -> dict:
    return {"stage": label, "rss_mb": round(ram_used_mb(), 1),
            "vram_device_mb": _r(vram_used_mb()), "vram_process_mb": _r(vram_process_mb())}


def _r(x):
    return round(x, 1) if x is not None else None


def fmt_buffers(name: str, bufs: dict) -> None:
    gpu, cpu = split_by_device(bufs)
    print(f"  {name}: accelerator {gpu:7.1f} MiB | CPU {cpu:7.1f} MiB")
    for dev, kinds in bufs.items():
        print(f"      {dev:<6} " + "  ".join(f"{k}={v:.1f}" for k, v in kinds.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-file")
    ap.add_argument("--n-ctx", type=int)
    ap.add_argument("--n-ubatch", type=int)
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--out", default="logs/memory.json")
    args = ap.parse_args()

    cfg = Config()
    if args.llm_file:
        cfg.llm.filename = args.llm_file
    if args.n_ctx:
        cfg.llm.n_ctx = args.n_ctx
    if args.n_ubatch:
        cfg.llm.n_ubatch = cfg.llm.n_batch = args.n_ubatch

    snaps = [snapshot("baseline")]
    with capture_stderr_fd() as emb_log:
        embedder = Embedder(cfg.embed, verbose=True)
    snaps.append(snapshot("embedder_loaded"))
    with capture_stderr_fd() as llm_log:
        llm = load_llm(cfg.llm, verbose=True)
    snaps.append(snapshot("llm_loaded"))
    llm.verbose = embedder.model.verbose = False  # silence per-request perf prints

    emb_bufs, llm_bufs = parse_llama_buffers(emb_log[0]), parse_llama_buffers(llm_log[0])
    chunks = load_chunks(cfg.chunks_path)
    retriever = Retriever(chunks, VectorIndex.load(cfg.index_path), embedder.embed_queries, cfg.retrieval)
    count = lambda t: len(llm.tokenize(t.encode(), add_bos=False, special=False))  # noqa: E731

    requests = []
    with PeakSampler() as peak:
        for q in args.queries:
            lang = detect_lang(q)
            ctx = retriever.select_context(retriever.retrieve(q))
            # message contents only (chat-template tokens add a few more)
            prompt_tokens = count("".join(m["content"] for m in build_messages(q, ctx, lang)))
            st = GenStats()
            for _ in measure_stream(stream_answer(llm, q, ctx, lang, cfg.llm.max_tokens,
                                                  cfg.llm.temperature, cfg.llm.repeat_penalty), st, count):
                pass
            requests.append({"query": q, "prompt_tokens": prompt_tokens, "output_tokens": st.n_tokens,
                             "ttft_ms": round(st.ttft_ms, 1), "decode_tps": round(st.decode_tps, 1)})
    snaps.append(snapshot("after_requests"))

    llm_gpu, llm_cpu = split_by_device(llm_bufs)
    emb_gpu, emb_cpu = split_by_device(emb_bufs)
    report = {
        "platform": f"{platform.system()} {platform.machine()}",
        "llm": cfg.llm.filename, "embed": cfg.embed.filename,
        "n_ctx": cfg.llm.n_ctx, "n_ubatch": cfg.llm.n_ubatch,
        "buffers_mib": {"llm": llm_bufs, "embedder": emb_bufs},
        "accelerator_total_mib": round(llm_gpu + emb_gpu, 1),
        "cpu_total_mib": round(llm_cpu + emb_cpu, 1),
        "snapshots": snaps,
        "peak_during_requests": {"rss_mb": round(peak.peak_rss_mb, 1), "vram_device_mb": _r(peak.peak_vram_mb),
                                 "samples": peak.samples},
        "requests": requests,
    }

    print(f"platform: {report['platform']}   LLM: {cfg.llm.filename}   n_ctx={cfg.llm.n_ctx} n_ubatch={cfg.llm.n_ubatch}")
    print("\nllama.cpp buffer accounting:")
    fmt_buffers("LLM     ", llm_bufs)
    fmt_buffers("Embedder", emb_bufs)
    print(f"  => accelerator (VRAM budget): {report['accelerator_total_mib']:.1f} MiB"
          f"   CPU RAM: {report['cpu_total_mib']:.1f} MiB")
    print("\nlive counters:")
    for s in snaps:
        print(f"  {s['stage']:<16} RSS {s['rss_mb']:8.1f} MB   VRAM(device) {s['vram_device_mb']}   VRAM(proc) {s['vram_process_mb']}")
    print(f"  peak during requests: RSS {report['peak_during_requests']['rss_mb']} MB, "
          f"VRAM(device) {report['peak_during_requests']['vram_device_mb']} ({peak.samples} samples)")
    print("\nrequests:")
    for r in requests:
        print(f"  {r['query'][:30]:<32} prompt={r['prompt_tokens']:>4} out={r['output_tokens']:>4} "
              f"TTFT={r['ttft_ms']:>6}ms decode={r['decode_tps']:>5} tok/s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
