"""Reproducible moderation evaluation.

    uv run python eval_moderation.py                 # full pipeline
    uv run python eval_moderation.py --input-only    # input guard only (no LLM needed)

Metrics:
  - sensitive block rate     = blocked sensitive / total sensitive
  - competitor block rate    = blocked competitor / total competitor
  - injection block rate     = blocked injection / total injection
  - false refusal rate       = refused allowed / total allowed
  - pass rate (normal)       = 1 - false refusal rate
  - moderation latency       = p50 / p95 ms per stage
Outputs a summary table + every failing case (no cherry-picking) to logs/eval_moderation.json.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--cases", default="tests/cases")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
