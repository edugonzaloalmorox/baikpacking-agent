from baikpacking.agents.event_resolution import _extract_event_name, resolve_event
from baikpacking.agents.recommender_agent import _extract_event_name as recommender_extract_event_name


KB_TITLES = (
    "Atlas Mountain Race",
    "Atlas Mountain Race 2024",
    "Tour Divide",
    "Pirenaica",
)


def _patch_kb_titles(monkeypatch, titles=KB_TITLES):
    from baikpacking.agents import event_resolution as mod

    monkeypatch.setattr(mod, "_load_kb_event_titles", lambda: tuple(titles))


def test_resolve_event_known_event_exact_title(monkeypatch):
    _patch_kb_titles(monkeypatch)

    result = resolve_event("Atlas Mountain Race")

    assert result.display_name == "Atlas Mountain Race"
    assert result.canonical_name == "Atlas Mountain Race"
    assert result.requested_count is None
    assert result.match_type == "trusted_exact"
    assert result.confidence >= 0.95
    assert result.is_trusted_exact is True
    assert result.candidate_events
    assert result.candidate_events[0].title == "Atlas Mountain Race"
    assert result.candidate_events[0].score >= 0.95


def test_resolve_event_alias(monkeypatch):
    _patch_kb_titles(monkeypatch)

    result = resolve_event("What lights should I use for AMR?")

    assert result.display_name == "Atlas Mountain Race"
    assert result.canonical_name == "Atlas Mountain Race"
    assert result.requested_count is None
    assert result.match_type == "trusted_exact"
    assert result.confidence >= 0.95
    assert result.is_trusted_exact is True
    assert result.candidate_events[0].title == "Atlas Mountain Race"
    assert result.candidate_events[0].source.startswith("alias:")


def test_resolve_event_unknown_low_confidence(monkeypatch):
    _patch_kb_titles(monkeypatch, titles=("Tour Divide",))

    result = resolve_event("North Cape Tarifa race")

    assert result.display_name == "North Cape Tarifa"
    assert result.canonical_name is None
    assert result.requested_count is None
    assert result.match_type == "unknown"
    assert result.confidence <= 0.35
    assert result.is_trusted_exact is False
    assert result.candidate_events
    assert result.candidate_events[0].score >= 0.0


def test_resolve_event_separates_requested_count_from_event_name(monkeypatch):
    _patch_kb_titles(monkeypatch)

    result = resolve_event("Give me 3 bikes suitable for the Pirenaica")

    assert result.requested_count == 3
    assert result.display_name == "Pirenaica"
    assert result.display_name != "3"
    assert result.match_type in {"trusted_exact", "fuzzy_candidate", "weak_candidate"}


def test_resolve_event_parses_count_with_known_event(monkeypatch):
    _patch_kb_titles(monkeypatch)

    result = resolve_event("Recommend 2 tyre options for Transiberica")

    assert result.requested_count == 2
    assert result.display_name == "Transiberica"
    assert result.canonical_name is None or result.canonical_name == "Transiberica"
    assert result.match_type in {"trusted_exact", "fuzzy_candidate", "weak_candidate"}


def test_resolve_event_parses_top_count_with_known_event(monkeypatch):
    _patch_kb_titles(monkeypatch)

    result = resolve_event("Top 5 bags for Atlas Mountain Race")

    assert result.requested_count == 5
    assert result.display_name == "Atlas Mountain Race"
    assert result.canonical_name == "Atlas Mountain Race"
    assert result.match_type == "trusted_exact"


def test_resolve_event_handles_count_when_no_named_event_exists(monkeypatch):
    _patch_kb_titles(monkeypatch, titles=("Tour Divide",))

    result = resolve_event("Give me 3 bikes for an ultra across Spain")

    assert result.requested_count == 3
    assert result.display_name != "3"
    assert result.display_name


def test_resolve_event_year_suffix_is_ignored(monkeypatch):
    _patch_kb_titles(monkeypatch, titles=("Atlas Mountain Race 2024",))

    result = resolve_event("Atlas Mountain Race")

    assert result.display_name == "Atlas Mountain Race"
    assert result.canonical_name == "Atlas Mountain Race"
    assert result.match_type == "trusted_exact"


def test_resolve_event_handles_noisy_event_phrase(monkeypatch):
    _patch_kb_titles(monkeypatch)

    result = resolve_event("What tyres should I use for the Atlas Mountain Race in Morocco?")

    assert result.display_name == "Atlas Mountain Race"
    assert result.canonical_name == "Atlas Mountain Race"
    assert result.match_type in {"trusted_exact", "fuzzy_candidate"}
    assert result.candidate_events
    assert result.candidate_events[0].title == "Atlas Mountain Race"


def test_resolve_event_display_name_matches_previous_extractor(monkeypatch):
    _patch_kb_titles(monkeypatch)

    queries = [
        "Atlas Mountain Race",
        "What lights should I use for AMR?",
        "North Cape Tarifa race",
        "Recommend a setup for Tour Divide 2024",
        "",
    ]

    for query in queries:
        assert resolve_event(query).display_name == _extract_event_name(query)
        assert resolve_event(query).display_name == recommender_extract_event_name(query)
