"""Interactive streaming chat.

    uv run aorus-chat                    # moderation on
    uv run aorus-chat --no-moderation    # baseline
    uv run aorus-chat --explain          # print moderation decisions and timings per turn
"""

from __future__ import annotations

import argparse
import sys

from aorus_assistant.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="AORUS MASTER 16 AM6H assistant")
    parser.add_argument("--no-moderation", action="store_true")
    parser.add_argument("--explain", action="store_true", help="show moderation decisions and timings")
    parser.add_argument("--llm-file", help="GGUF filename in models/ (overrides config)")
    args = parser.parse_args()

    cfg = Config()
    if args.llm_file:
        cfg.llm.filename = args.llm_file
    print("Loading models...", file=sys.stderr)
    from aorus_assistant.pipeline import AssistantPipeline

    pipe = AssistantPipeline(cfg, moderation=not args.no_moderation)
    mode = "off" if args.no_moderation else "on"
    print(f"AORUS MASTER 16 AM6H assistant (moderation {mode}). Ctrl-D to exit.", file=sys.stderr)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        res = pipe.answer(query, on_token=lambda t: print(t, end="", flush=True))
        print()
        if args.explain:
            for d in res.decisions:
                if d:
                    print(f"  [{d.stage}] {d.category.value}/{d.action.value}: {d.reason}", file=sys.stderr)
            t = res.timings_ms
            print("  timings ms: " + ", ".join(f"{k}={v:.0f}" for k, v in t.items()), file=sys.stderr)
            if res.contexts:
                print(f"  contexts: {[rc.chunk.id for rc in res.contexts]}", file=sys.stderr)


if __name__ == "__main__":
    main()
