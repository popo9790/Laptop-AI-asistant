"""Sanity checks on the test-case files themselves."""

VALID_CATEGORIES = {"allowed", "sensitive", "off_topic", "competitor", "prompt_injection", "no_relevant_data"}
VALID_LANGS = {"zh", "en", "mixed"}


def test_cases_well_formed(all_cases):
    ids = [c["id"] for c in all_cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in all_cases:
        assert c["query"].strip()
        assert c["category"] in VALID_CATEGORIES, c
        assert c["lang"] in VALID_LANGS, c
