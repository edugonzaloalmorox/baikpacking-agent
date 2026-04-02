from contextlib import nullcontext
from types import SimpleNamespace

from baikpacking.agents.models import ChunkInfo, QueryIntent, SetupCore, SetupRecommendation, SimilarRider
from baikpacking.agents.orchestration_models import EventContextSummary, EventResolutionResult, RetrievalPlan


def _rider(**kwargs):
    defaults = dict(
        rider_id=1,
        best_score=0.91,
        event_title="Pirenaica 2025",
        tyres="45mm semi-slick",
        chunks=[ChunkInfo(score=0.8, text="45mm semi-slick tyres", chunk_index=None)],
    )
    defaults.update(kwargs)
    return SimilarRider(**defaults)


def _patch_runtime(monkeypatch, mod, writer_run_sync):
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
    riders = [_rider()]
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
    monkeypatch.setattr(mod.writer_agent, "run_sync", writer_run_sync)
    return writer_output


def test_recommender_uses_one_writer_call_on_happy_path(monkeypatch):
    from baikpacking.agents import recommender_agent as mod

    writer_output = SetupRecommendation(
        summary="Grounded summary",
        reasoning="Grounded reasoning",
        recommended_setup=SetupCore(),
        similar_riders=[],
    )

    def writer_run_sync(_prompt):
        return SimpleNamespace(output=writer_output)

    _patch_runtime(monkeypatch, mod, writer_run_sync)

    rec, trace = mod.recommend_setup_with_trace("Give me tyres for Pirenaica")

    assert isinstance(rec, SetupRecommendation)
    summary_step = next(call for call in trace.calls if call["tool"] == "writer_stage_summary")
    assert summary_step["result"]["writer_call_count"] == 1
    assert summary_step["result"]["writer_second_pass_triggered"] is False
    assert summary_step["result"]["writer_second_pass_reason"] is None


def test_recommender_records_explicit_repair_pass(monkeypatch):
    from baikpacking.agents import recommender_agent as mod

    writer_output = SetupRecommendation(
        summary="Grounded summary",
        reasoning="Grounded reasoning",
        recommended_setup=SetupCore(),
        similar_riders=[],
    )

    calls = {"n": 0}

    def writer_run_sync(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("invalid writer output")
        return SimpleNamespace(output=writer_output)

    _patch_runtime(monkeypatch, mod, writer_run_sync)

    rec, trace = mod.recommend_setup_with_trace("Give me tyres for Pirenaica")

    assert isinstance(rec, SetupRecommendation)
    summary_step = next(call for call in trace.calls if call["tool"] == "writer_stage_summary")
    assert summary_step["result"]["writer_call_count"] == 2
    assert summary_step["result"]["writer_second_pass_triggered"] is True
    assert "RuntimeError" in summary_step["result"]["writer_second_pass_reason"]
