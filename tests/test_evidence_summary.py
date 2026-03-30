from contextlib import nullcontext
from types import SimpleNamespace

from baikpacking.agents.evidence_summary import _rider_component_hit_count, summarize_evidence
from baikpacking.agents.models import ChunkInfo, QueryIntent, SetupCore, SetupRecommendation, SimilarRider
from baikpacking.agents.orchestration_models import (
    EventContextSummary,
    EventResolutionResult,
    RetrievalExecutionResult,
    RetrievalPlan,
)


def _rider(**kwargs):
    defaults = dict(
        bike_type=None,
        wheels=None,
        tyres=None,
        drivetrain=None,
        bags=None,
        sleep_system=None,
        key_items=[],
        chunks=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_rider_count_and_component_hit_count():
    riders = [
        _rider(tyres="45mm"),
        _rider(tyres="40mm", chunks=[SimpleNamespace(text="tubeless tyre setup")]),
    ]
    intent = QueryIntent(component="tyres", confidence=0.5, component_terms=["tyre", "tyres"])
    event_resolution = EventResolutionResult(display_name="Atlas Mountain Race", is_trusted_exact=True)
    retrieval_result = RetrievalExecutionResult(riders=riders, used_query="q")

    summary = summarize_evidence(riders, intent, event_resolution, retrieval_result)

    assert summary.rider_count == 2
    assert summary.component_hit_count == _rider_component_hit_count(riders, intent.component_terms)


def test_field_support_classification_is_deterministic():
    riders = [
        _rider(tyres="45mm", bags="frame bag", chunks=[SimpleNamespace(text="bikepacking bags")]),
        _rider(tyres="40mm", bags="seat pack"),
        _rider(tyres=None, bags=None, chunks=[SimpleNamespace(text="sleep system bivy")]),
    ]
    intent = QueryIntent(component="full_setup", confidence=0.25, component_terms=[])
    event_resolution = EventResolutionResult(display_name="Unknown event", is_trusted_exact=False)
    retrieval_result = RetrievalExecutionResult(riders=riders, used_query="q")

    summary = summarize_evidence(riders, intent, event_resolution, retrieval_result)

    assert summary.field_support["tyres"] in {"weak_pattern", "pattern"}
    assert summary.field_support["bags"] in {"weak_pattern", "pattern"}
    assert summary.field_support["sleep_system"] == "single"
    assert summary.field_support["drivetrain"] == "none"


def test_evidence_strength_and_consistency_labels():
    riders = [
        _rider(tyres="45mm", bags="frame bag", wheels="700c"),
        _rider(tyres="45mm", bags="seat pack", wheels="700c"),
        _rider(tyres="45mm", bags="frame bag", wheels="700c"),
        _rider(tyres="45mm", bags="frame bag", wheels="700c"),
    ]
    intent = QueryIntent(component="tyres", confidence=0.5, component_terms=["tyre"])
    event_resolution = EventResolutionResult(display_name="Tour Divide", is_trusted_exact=True)
    retrieval_result = RetrievalExecutionResult(riders=riders, used_query="q")

    summary = summarize_evidence(riders, intent, event_resolution, retrieval_result)

    assert summary.evidence_strength in {"moderate", "strong"}
    assert summary.consistency in {"consistent", "mostly_consistent"}


def test_recommendation_behavior_regression(monkeypatch):
    from baikpacking.agents import recommender_agent as mod

    event_resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="exact",
        confidence=0.99,
        is_trusted_exact=True,
    )
    intent = QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"])
    event_context = EventContextSummary(
        requested_event_name="Atlas Mountain Race",
        web_context_text="ultra-distance mountain race",
    )
    retrieval_plan = RetrievalPlan(
        query_component="tyres",
        use_exact_event=True,
        event_name_for_retrieval="Atlas Mountain Race",
        descriptor_query="atlas mountain race tyres",
        descriptor_query_with_intent=None,
        primary_query="atlas mountain race tyres",
        fallback_query=None,
        fallback_reasoning=None,
        intent_bundle=None,
    )
    riders = [
        SimilarRider(
            rider_id=1,
            best_score=0.91,
            event_title="Atlas Mountain Race",
            tyres="45mm",
            chunks=[ChunkInfo(score=0.8, text="tubeless tyre setup", chunk_index=None)],
        )
    ]
    writer_output = SetupRecommendation(
        summary="Grounded summary",
        reasoning="Grounded reasoning",
        recommended_setup=SetupCore(),
        similar_riders=[],
    )

    monkeypatch.setattr(mod, "logfire", SimpleNamespace(span=lambda *args, **kwargs: nullcontext()))
    monkeypatch.setattr(mod, "_build_deps", lambda call_trace=None: SimpleNamespace(call_trace=call_trace))
    monkeypatch.setattr(mod, "time_and_record", lambda *args, fn=None, **kwargs: fn())
    monkeypatch.setattr(mod, "resolve_event", lambda user_query: event_resolution)
    monkeypatch.setattr(mod, "_classify_query_intent", lambda user_query: intent)
    monkeypatch.setattr(mod, "fetch_event_context_summary", lambda event_resolution, deps: event_context)
    monkeypatch.setattr(mod, "build_retrieval_plan", lambda **kwargs: retrieval_plan)
    monkeypatch.setattr(mod, "run_search_similar_riders", lambda **kwargs: riders)
    monkeypatch.setattr(mod.writer_agent, "run_sync", lambda payload: SimpleNamespace(output=writer_output))

    rec, trace = mod.recommend_setup_with_trace("What tyres should I use for Atlas Mountain Race?")

    assert trace is not None
    assert rec.summary == "Grounded summary"
    assert rec.event == "Atlas Mountain Race"
    assert rec.tyres == "45mm"
    assert rec.similar_riders[0].event_title == "Atlas Mountain Race"
