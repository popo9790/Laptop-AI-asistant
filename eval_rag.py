"""RAG quality evaluation.

    uv run python eval_rag.py --retrieval-only     # Recall@k / MRR, dense vs BM25 vs hybrid
    uv run python eval_rag.py                      # + answer correctness, no-data refusal (TODO)

Retrieval metrics on tests/cases/retrieval.jsonl (query -> gold chunk ids):
  Hit@k  = fraction of queries with >=1 gold chunk in the top k
  MRR    = mean of 1 / rank of the first gold chunk (0 if not in top 10)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aorus_assistant.config import Config
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


def eval_retrieval(retriever: Retriever, cases: list[dict], weights: dict[str, float]) -> dict:
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


def build_retriever(cfg: Config) -> Retriever:
    chunks = load_chunks(cfg.chunks_path)
    index = VectorIndex.load(cfg.index_path)
    embedder = Embedder(cfg.embed)
    return Retriever(chunks, index, embedder.embed_queries, cfg.retrieval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--out", default="logs/eval_retrieval.json")
    args = parser.parse_args()

    cfg = Config()
    retriever = build_retriever(cfg)
    weights = {"bm25": 0.0, "dense": 1.0, **{f"hybrid_w{w}": w for w in (0.3, 0.5, 0.6, 0.7)}}
    report = eval_retrieval(retriever, load_retrieval_cases(), weights)
    report["embed_model"] = retriever.index.model_name

    print(f"embed model: {report['embed_model']}   n={report['n']}   "
          f"query latency p50={report['query_latency_ms']['p50']}ms p95={report['query_latency_ms']['p95']}ms")
    print(f"{'mode':<14}" + "".join(f"{k:>9}" for k in [*(f'hit@{k}' for k in KS), 'mrr']))
    for mode, s in report["summary"].items():
        print(f"{mode:<14}" + "".join(f"{v:>9.3f}" for v in s.values()))
    for mode in ("dense", "bm25", f"hybrid_w{cfg.retrieval.dense_weight}"):
        for f in report["failures"].get(mode, []):
            print(f"  [miss@1 {mode}] {f['id']} {f['query']!r} gold={f['gold'][:2]} got={f['top3']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.retrieval_only:
        print("answer-level evaluation not implemented yet")


if __name__ == "__main__":
    main()
