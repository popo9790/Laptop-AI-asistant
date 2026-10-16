"""RAG quality evaluation on tests/cases/retrieval.jsonl.

    uv run python eval_rag.py --retrieval-only           # Hit@k / MRR, dense vs BM25 vs hybrid
    uv run python eval_rag.py                            # + answer-level metrics (needs LLM)
    uv run python eval_rag.py --llm-file Qwen3.5-0.8B-Q8_0.gguf   # compare quantisations

Retrieval (answerable cases only):
  Hit@k  = fraction of queries with >=1 gold chunk in the top k
  MRR    = mean of 1 / rank of the first gold chunk (0 if not in top 10)
Answer level:
  answer accuracy  = answerable cases whose answer contains every `answer` group
                     (any alternative within a group) and is not a refusal
  false refusal    = answerable cases the system refused
  no-data refusal  = unanswerable (`expect_refusal`) cases correctly refused, either by the
                     retrieval threshold (before the LLM) or by the LLM itself
  simplified-zh    = zh answers containing Simplified-only characters (should be 0)
Substring matching is strict but cheap and reproducible; failures are printed in full so
they can be inspected manually rather than trusted blindly.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from aorus_assistant.bench.timing import GenStats, measure_stream
from aorus_assistant.config import Config
from aorus_assistant.llm.generator import stream_answer
from aorus_assistant.llm.models import load_llm
from aorus_assistant.moderation.responses import detect_lang, refusal_text
from aorus_assistant.rag.embedder import Embedder
from aorus_assistant.rag.index import VectorIndex
from aorus_assistant.rag.retriever import Retriever, load_chunks

CASES = Path("tests/cases/retrieval.jsonl")
KS = (1, 3, 5)


def load_retrieval_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_metrics(ranked_ids: list[str], gold: set[str]) -> dict[str, float]:
    first = next((i + 1 for i, cid in enumerate(ranked_ids[:10]) if cid in gold), None)
    out = {f"hit@{k}": float(first is not None and first <= k) for k in KS}
    out["mrr"] = 1.0 / first if first else 0.0
    return out


REFUSAL_RE = re.compile(
    r"沒有.{0,20}(資訊|資料|說明|數據)|無法.{0,4}(回答|提供|確定)|未.{0,4}(提供|標示|列出|提及|說明)"
    r"|don't have|do not have|not available|no information|not (specified|mentioned|listed)"
    r"|does(?: not|n't) (contain|specify|include|mention)",
    re.IGNORECASE,
)
# Characters that exist only in Simplified Chinese (their Traditional forms differ).
SIMPLIFIED_ONLY = set("这个们为显处来时对说没还开关发电总线么样从无连储设备键盘频费适应择问题帮")


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text))


def matches_answer(text: str, groups: list[list[str]]) -> bool:
    low = text.lower()
    return all(any(alt.lower() in low for alt in group) for group in groups)


def eval_retrieval(retriever: Retriever, cases: list[dict], weights: dict[str, float]) -> dict:
    cases = [c for c in cases if c["gold"]]
    ids = retriever.index.ids
    per_mode: dict[str, list[dict]] = {m: [] for m in weights}
    failures: dict[str, list[dict]] = {m: [] for m in weights}
    latencies = []
    for case in cases:
        t0 = time.perf_counter()
        dense, sparse = retriever.score(case["query"])
        latencies.append((time.perf_counter() - t0) * 1000)
        sparse_n = sparse / sparse.max() if sparse.max() > 0 else sparse
        gold = set(case["gold"])
        for mode, w in weights.items():
            hybrid = w * dense + (1 - w) * sparse_n
            ranked = [ids[i] for i in np.argsort(-hybrid)]
            m = rank_metrics(ranked, gold)
            per_mode[mode].append(m)
            if m["hit@1"] == 0:
                failures[mode].append(
                    {"id": case["id"], "query": case["query"], "gold": case["gold"], "top3": ranked[:3]}
                )
    summary = {
        mode: {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
        for mode, rows in per_mode.items()
    }
    return {
        "n": len(cases),
        "summary": summary,
        "failures": failures,
        "query_latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
        },
    }


def eval_answers(retriever: Retriever, llm, cfg: Config, cases: list[dict]) -> dict:
    count = lambda t: len(llm.tokenize(t.encode(), add_bos=False, special=False))  # noqa: E731
    rows = []
    for case in cases:
        q, lang = case["query"], detect_lang(case["query"])
        results = retriever.retrieve(q)
        stats = GenStats()
        if not retriever.has_relevant(results):
            answer, refused_by = refusal_text(lang), "retrieval_threshold"
        else:
            ctx = retriever.select_context(results)
            stream = stream_answer(
                llm, q, ctx, lang, cfg.llm.max_tokens, cfg.llm.temperature, cfg.llm.repeat_penalty
            )
            answer = "".join(measure_stream(stream, stats, count))
            refused_by = "llm" if is_refusal(answer) else None
        expect_refusal = case.get("expect_refusal", False)
        correct = (refused_by is not None) if expect_refusal else (
            refused_by is None and matches_answer(answer, case["answer"])
        )
        rows.append({
            "id": case["id"], "query": q, "lang": case["lang"], "answer": answer,
            "expect_refusal": expect_refusal, "refused_by": refused_by, "correct": correct,
            "top_dense": round(results[0].dense_score, 3) if results else None,
            "simplified": lang == "zh" and bool(SIMPLIFIED_ONLY & set(answer)),
            "ttft_ms": round(stats.ttft_ms, 1), "decode_tps": round(stats.decode_tps, 1),
            "n_tokens": stats.n_tokens,
        })

    answerable = [r for r in rows if not r["expect_refusal"]]
    unanswerable = [r for r in rows if r["expect_refusal"]]
    generated = [r for r in rows if r["n_tokens"] > 0]
    zh = [r for r in rows if detect_lang(r["query"]) == "zh"]
    mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None  # noqa: E731
    return {
        "summary": {
            "n_answerable": len(answerable),
            "answer_accuracy": mean([r["correct"] for r in answerable]),
            "false_refusal_rate": mean([r["refused_by"] is not None for r in answerable]),
            "n_unanswerable": len(unanswerable),
            "no_data_refusal_rate": mean([r["correct"] for r in unanswerable]),
            "simplified_zh_rate": mean([r["simplified"] for r in zh]),
            "ttft_ms_p50": mean([np.median([r["ttft_ms"] for r in generated])]) if generated else None,
            "decode_tps_mean": mean([r["decode_tps"] for r in generated if r["n_tokens"] > 1]),
            "by_lang": {
                lg: mean([r["correct"] for r in answerable if r["lang"] == lg])
                for lg in ("zh", "en", "mixed")
            },
        },
        "rows": rows,
    }


def build_retriever(cfg: Config) -> Retriever:
    chunks = load_chunks(cfg.chunks_path)
    index = VectorIndex.load(cfg.index_path)
    embedder = Embedder(cfg.embed)
    return Retriever(chunks, index, embedder.embed_queries, cfg.retrieval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--llm-file", help="GGUF filename in models/ (overrides config)")
    parser.add_argument("--out", default="logs/eval_rag.json")
    args = parser.parse_args()

    cfg = Config()
    if args.llm_file:
        cfg.llm.filename = args.llm_file
    retriever = build_retriever(cfg)
    weights = {"bm25": 0.0, "dense": 1.0, **{f"hybrid_w{w}": w for w in (0.3, 0.5, 0.6, 0.7)}}
    cases = load_retrieval_cases()
    report = eval_retrieval(retriever, cases, weights)
    report["embed_model"] = retriever.index.model_name

    print(f"embed model: {report['embed_model']}   n={report['n']}   "
          f"query latency p50={report['query_latency_ms']['p50']}ms p95={report['query_latency_ms']['p95']}ms")
    print(f"{'mode':<14}" + "".join(f"{k:>9}" for k in [*(f'hit@{k}' for k in KS), 'mrr']))
    for mode, s in report["summary"].items():
        print(f"{mode:<14}" + "".join(f"{v:>9.3f}" for v in s.values()))
    for mode in ("dense", "bm25", f"hybrid_w{cfg.retrieval.dense_weight}"):
        for f in report["failures"].get(mode, []):
            print(f"  [miss@1 {mode}] {f['id']} {f['query']!r} gold={f['gold'][:2]} got={f['top3']}")

    if not args.retrieval_only:
        llm = load_llm(cfg.llm)
        ans = eval_answers(retriever, llm, cfg, cases)
        report["answers"] = ans
        report["llm_model"] = cfg.llm.filename
        print(f"\nLLM: {cfg.llm.filename}")
        for k, v in ans["summary"].items():
            print(f"  {k:<22} {v}")
        for r in ans["rows"]:
            if not r["correct"] or r["simplified"]:
                tag = "MISS-REFUSE" if r["expect_refusal"] else ("FALSE-REFUSE" if r["refused_by"] else "WRONG")
                if r["correct"]:
                    tag = "SIMPLIFIED"
                print(f"  [{tag}] {r['id']} {r['query']!r} (top_dense={r['top_dense']})\n      -> {r['answer'][:200]!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
