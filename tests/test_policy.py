from contextlib import nullcontext
from types import SimpleNamespace

from baikpacking.agents.evidence_summary import summarize_evidence
from baikpacking.agents.models import ChunkInfo, QueryIntent, SetupCore, SetupRecommendation, SimilarRider
from baikpacking.agents.output_validation import _fill_requested_component_from_riders
from baikpacking.agents.orchestration_models import EventContextSummary, EventResolutionResult, RetrievalExecutionResult, RetrievalPlan
from baikpacking.agents.policy import select_policy


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


def test_strong_evidence_allows_specific_examples():
    intent = QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"])
    riders = [
        _rider(tyres=None, chunks=[SimpleNamespace(text="45mm semi-slick tyres")]),
        _rider(tyres=None, chunks=[SimpleNamespace(text="40mm tubeless tyres")]),
        _rider(tyres=None, chunks=[SimpleNamespace(text="42mm fast-rolling tyres")]),
        _rider(tyres=None, chunks=[SimpleNamespace(text="45mm semi-slick tyres")]),
    ]
    event_resolution = EventResolutionResult(display_name="Pirenaica", match_type="unknown", is_trusted_exact=False)
    evidence = summarize_evidence(
        riders,
        intent,
        event_resolution,
        RetrievalExecutionResult(riders=riders, used_query="q"),
    )
    policy = select_policy(
        event_match_type="unknown",
        matched_event_name="Pirenaica 2025",
        retrieval_source="exact_event",
        exact_event_hit_count=4,
        evidence_strength=evidence.evidence_strength,
    )

    out = _fill_requested_component_from_riders(
        rec=SetupRecommendation(summary="x", reasoning="y"),
        riders=riders,
        query_component="tyres",
        policy=policy,
    )

    assert policy.mode == "strict_grounded"
    assert policy.allow_event_specific_claims is True
    assert "unknown_event" not in policy.notes
    assert out.recommended_setup.tyres == "45mm semi-slick tyres"
    assert "Grounded examples for tyres" in (out.recommended_setup.notes or "")
    assert "40mm tubeless tyres" in (out.recommended_setup.notes or "")


def test_weak_evidence_prefers_generic_and_avoids_chunk_backfill():
    policy = select_policy(
        event_match_type="unknown",
        matched_event_name=None,
        retrieval_source="unknown_global",
        exact_event_hit_count=0,
        evidence_strength="weak",
    )

    out = _fill_requested_component_from_riders(
        rec=SetupRecommendation(summary="x", reasoning="y"),
        riders=[_rider(tyres=None, chunks=[SimpleNamespace(text="45mm semi-slick tyres")])],
        query_component="tyres",
        policy=policy,
    )

    assert policy.mode == "generic_fallback"
    assert policy.allow_event_specific_claims is False
    assert out.recommended_setup.tyres is None
    assert out.recommended_setup.notes is None


def test_unknown_event_restricts_event_specific_claims():
    known_event = EventResolutionResult(display_name="Atlas Mountain Race", is_trusted_exact=True)
    unknown_event = EventResolutionResult(display_name="Unknown event", is_trusted_exact=False)
    riders = [
        _rider(tyres="45mm semi-slick"),
        _rider(tyres="40mm tubeless"),
        _rider(tyres="42mm fast-rolling"),
    ]

    known_policy = select_policy(
        event_match_type=known_event.match_type,
        matched_event_name="Atlas Mountain Race",
        retrieval_source="exact_event",
        exact_event_hit_count=3,
        evidence_strength="strong",
    )
    unknown_policy = select_policy(
        event_match_type=unknown_event.match_type,
        matched_event_name=None,
        retrieval_source="unknown_global",
        exact_event_hit_count=0,
        evidence_strength="strong",
    )

    known_out = _fill_requested_component_from_riders(
        rec=SetupRecommendation(summary="x", reasoning="y"),
        riders=riders,
        query_component="tyres",
        policy=known_policy,
    )
    unknown_out = _fill_requested_component_from_riders(
        rec=SetupRecommendation(summary="x", reasoning="y"),
        riders=riders,
        query_component="tyres",
        policy=unknown_policy,
    )

    assert known_policy.allow_event_specific_claims is True
    assert unknown_policy.allow_event_specific_claims is False
    assert known_out.recommended_setup.notes is not None
    assert unknown_out.recommended_setup.notes is None


def test_similar_event_medium_evidence_uses_pattern_based():
    policy = select_policy(
        event_match_type="unknown",
        matched_event_name="Transcontinental Race",
        retrieval_source="similar_event",
        exact_event_hit_count=0,
        evidence_strength="moderate",
    )

    assert policy.mode == "pattern_based"
    assert policy.allow_event_specific_claims is False
    assert "similar_event_retrieval" in policy.notes


def test_pirenaica_exact_retrieval_overrides_unknown_parser_match(monkeypatch):
    from baikpacking.agents import recommender_agent as mod

    event_resolution = EventResolutionResult(
        display_name="Pirenaica",
        canonical_name=None,
        match_type="unknown",
        confidence=0.15,
        is_trusted_exact=False,
    )
    intent = QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"])
    event_context = EventContextSummary(
        requested_event_name="Pirenaica",
        web_context_text="ultra-distance gravel race in Spain",
        similar_events=[],
        event_family=None,
        family_confidence=0.0,
        archetype=None,
        surface_family=None,
        features={},
    )
    retrieval_plan = RetrievalPlan(
        query_component="tyres",
        use_exact_event=True,
        event_name_for_retrieval="Pirenaica",
        descriptor_query="pirenaica tyres",
        descriptor_query_with_intent=None,
        primary_query="pirenaica tyres",
        fallback_query=None,
        fallback_reasoning=None,
        intent_bundle=None,
    )
    riders = [
        SimilarRider(
            rider_id=1,
            best_score=0.91,
            event_title="Pirenaica 2025",
            tyres="45mm semi-slick",
            chunks=[ChunkInfo(score=0.8, text="tubeless tyre setup", chunk_index=None)],
        ),
        SimilarRider(
            rider_id=2,
            best_score=0.89,
            event_title="Pirenaica 2025",
            tyres="40mm tubeless",
            chunks=[ChunkInfo(score=0.8, text="40mm tubeless tyres", chunk_index=None)],
        ),
        SimilarRider(
            rider_id=3,
            best_score=0.88,
            event_title="Pirenaica 2025",
            tyres="42mm fast-rolling",
            chunks=[ChunkInfo(score=0.8, text="42mm fast-rolling tyres", chunk_index=None)],
        ),
        SimilarRider(
            rider_id=4,
            best_score=0.87,
            event_title="Pirenaica 2025",
            tyres="45mm semi-slick",
            chunks=[ChunkInfo(score=0.8, text="45mm semi-slick tyres", chunk_index=None)],
        ),
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

    rec, trace = mod.recommend_setup_with_trace("Give me 3 bikes suitable for the Pirenaica")

    policy_calls = [call for call in trace.calls if call["tool"] == "policy_selection"]
    assert policy_calls, "expected a policy_selection trace entry"
    policy_result = policy_calls[-1]["result"]

    assert policy_result["mode"] == "strict_grounded"
    assert "unknown_event" not in policy_result["notes"]
    assert policy_result["allow_event_specific_claims"] is True
    assert rec.summary == "Grounded summary"
    assert rec.tyres == "45mm semi-slick"
