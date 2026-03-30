from baikpacking.agents.event_context_resolution import (
    _build_descriptor_query,
    fetch_event_context_summary,
)
from baikpacking.agents.orchestration_models import EventResolutionResult


class _DummyCtx:
    def __init__(
        self,
        summary="Remote ultra",
        surface="Mostly paved",
        route_character="Europe crossing",
        climate_notes="Windy",
        resupply_notes="Sparse at night",
        constraints=None,
    ):
        self.summary = summary
        self.surface = surface
        self.route_character = route_character
        self.climate_notes = climate_notes
        self.resupply_notes = resupply_notes
        self.constraints = constraints or []


class _DummyEventContextObj:
    def __init__(self, context):
        self.context = context


def test_descriptor_query_known_event_regression_shape():
    descriptor = _build_descriptor_query(
        event_name="Atlas Mountain Race",
        event_context="mountainous remote 1200 km 20000 m climbing",
        user_question="What tyres should I use for Atlas Mountain Race?",
    )

    assert descriptor["archetype"]
    assert descriptor["surface_family"]
    assert "Atlas Mountain Race" in descriptor["descriptor_query"]
    assert "1200 km" in descriptor["descriptor_query"]


def test_fetch_event_context_summary_known_event(monkeypatch):
    from baikpacking.agents import event_context_resolution as mod

    monkeypatch.setattr(
        mod,
        "run_event_web_search_sync",
        lambda event_title, deps: _DummyEventContextObj(
            _DummyCtx(
                summary="Remote mountain ultra",
                surface="Off-road",
                route_character="Long climbs",
            )
        ),
    )

    resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="exact",
        confidence=0.98,
        is_trusted_exact=True,
    )

    summary = fetch_event_context_summary(resolution, deps=object())

    assert summary.requested_event_name == "Atlas Mountain Race"
    assert summary.archetype is not None
    assert summary.surface_family in {"mixed_offroad", "mtb", "gravel", "road", "unknown"}
    assert summary.family_confidence >= 0.8


def test_fetch_event_context_summary_unknown_event_low_confidence(monkeypatch):
    from baikpacking.agents import event_context_resolution as mod

    monkeypatch.setattr(
        mod,
        "run_event_web_search_sync",
        lambda event_title, deps: _DummyEventContextObj(
            _DummyCtx(
                summary="Ultra-distance Europe crossing race",
                surface="Mostly paved",
                route_character="Across Europe",
            )
        ),
    )

    resolution = EventResolutionResult(
        display_name="North Cape Tarifa",
        canonical_name=None,
        match_type="unknown",
        confidence=0.15,
        is_trusted_exact=False,
    )

    summary = fetch_event_context_summary(resolution, deps=object())

    assert summary.event_family == "trans-europe ultra"
    assert summary.family_confidence <= 0.35


def test_fetch_event_context_summary_sparse_context_low_confidence(monkeypatch):
    from baikpacking.agents import event_context_resolution as mod

    monkeypatch.setattr(
        mod,
        "run_event_web_search_sync",
        lambda event_title, deps: _DummyEventContextObj(
            _DummyCtx(summary="", surface="", route_character="", climate_notes="", resupply_notes="")
        ),
    )

    resolution = EventResolutionResult(
        display_name="Unknown event",
        canonical_name=None,
        match_type="unknown",
        confidence=0.15,
        is_trusted_exact=False,
    )

    summary = fetch_event_context_summary(resolution, deps=object())

    assert summary.web_context_text == ""
    assert summary.family_confidence <= 0.1
