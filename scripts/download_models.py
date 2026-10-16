"""Download the GGUF LLM + embedding model into models/.

    uv run python scripts/download_models.py
"""

from aorus_assistant.config import Config
from aorus_assistant.llm.models import ensure_model


def main() -> None:
    cfg = Config()
    for repo_id, filename in [(cfg.llm.repo_id, cfg.llm.filename), (cfg.embed.repo_id, cfg.embed.filename)]:
        print(ensure_model(repo_id, filename))


if __name__ == "__main__":
    main()
