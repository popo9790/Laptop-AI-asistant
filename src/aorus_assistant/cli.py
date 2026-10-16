"""Interactive streaming chat: `uv run aorus-chat`."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="AORUS MASTER 16 AM6H assistant")
    parser.add_argument("--no-moderation", action="store_true")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
