import importlib

import pytest

MODULES = [
    "aorus_assistant.config",
    "aorus_assistant.schemas",
    "aorus_assistant.pipeline",
    "aorus_assistant.rag.scraper",
    "aorus_assistant.rag.parser",
    "aorus_assistant.rag.chunker",
    "aorus_assistant.rag.index",
    "aorus_assistant.rag.retriever",
    "aorus_assistant.llm.generator",
    "aorus_assistant.moderation.input_guard",
    "aorus_assistant.moderation.retrieval_filter",
    "aorus_assistant.moderation.output_guard",
    "aorus_assistant.moderation.audit",
    "aorus_assistant.bench.timing",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
