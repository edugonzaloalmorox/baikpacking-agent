from contextlib import nullcontext
from types import SimpleNamespace

from baikpacking.agents.models import ChunkInfo, QueryIntent, SetupCore, SetupRecommendation, SimilarRider, WriterRecommendationDraft
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


def test_writer_contract_is_minimal():
    assert set(WriterRecommendationDraft.model_fields) == {"event", "summary", "reasoning", "recommended_setup"}


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


def test_recommender_keeps_exact_event_scope_from_being_overwritten(monkeypatch):
    from baikpacking.agents import recommender_agent as mod

    exact_rider = _rider(event_title="B-HARD Ultra Race and Brevet 2023", tyres="45mm semi-slick")
    setattr(exact_rider, "_source_scope", "exact_event")

    calls = {"n": 0}

    def run_search_similar_riders(**kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("fallback retrieval should not run when exact scope is present")
        return [exact_rider]

    def writer_run_sync(_prompt):
        return SimpleNamespace(
            output=SetupRecommendation(
                summary="Grounded summary",
                reasoning="Grounded reasoning",
                recommended_setup=SetupCore(),
                similar_riders=[],
            )
        )

    event_resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="alias",
        confidence=0.93,
        is_trusted_exact=False,
    )
    intent = QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"])
    event_context = EventContextSummary(
        requested_event_name="Atlas Mountain Race",
        web_context_text="remote mountain ultra",
        similar_events=["GB Duro"],
        event_family="mountain_gravel_ultra",
        family_confidence=0.5,
        archetype="mountain_offroad_ultra",
        surface_family="mixed_offroad",
        features={},
    )
    retrieval_plan = RetrievalPlan(
        query_component="tyres",
        use_exact_event=False,
        event_name_for_retrieval="Atlas Mountain Race",
        descriptor_query="atlas mountain race tyres",
        descriptor_query_with_intent="atlas mountain race tyres focus",
        primary_query="atlas mountain race tyres",
        fallback_query="atlas mountain race tyres focus",
        fallback_reasoning="component_specific_then_broaden",
        intent_bundle=None,
    )

    monkeypatch.setattr(mod, "logfire", SimpleNamespace(span=lambda *args, **kwargs: nullcontext()))
    monkeypatch.setattr(mod, "_build_deps", lambda call_trace=None: SimpleNamespace(call_trace=call_trace))
    monkeypatch.setattr(mod, "time_and_record", lambda *args, fn=None, **kwargs: fn())
    monkeypatch.setattr(mod, "resolve_event", lambda user_query: event_resolution)
    monkeypatch.setattr(mod, "_classify_query_intent", lambda user_query: intent)
    monkeypatch.setattr(mod, "fetch_event_context_summary", lambda event_resolution, deps: event_context)
    monkeypatch.setattr(mod, "build_retrieval_plan", lambda **kwargs: retrieval_plan)
    monkeypatch.setattr(mod, "run_search_similar_riders", run_search_similar_riders)
    monkeypatch.setattr(
        mod.writer_agent,
        "run_sync",
        writer_run_sync,
    )

    rec, trace = mod.recommend_setup_with_trace("What tyres should I use for AMR?")

    assert isinstance(rec, SetupRecommendation)
    policy_step = next(call for call in trace.calls if call["tool"] == "policy_selection")
    assert policy_step["args"]["retrieval_source"] == "exact_event"
    assert policy_step["args"]["exact_event_hit_count"] == 1


def test_recommender_injects_review_context_into_writer_prompt(monkeypatch):
    from baikpacking.agents import recommender_agent as mod

    prompt_holder = {"prompt": None}

    def writer_run_sync(prompt):
        prompt_holder["prompt"] = prompt
        return SimpleNamespace(
            output=SetupRecommendation(
                summary="Grounded summary",
                reasoning="Grounded reasoning",
                recommended_setup=SetupCore(),
                similar_riders=[],
            )
        )

    _patch_runtime(monkeypatch, mod, writer_run_sync)
    monkeypatch.setattr(
        mod,
        "load_reviews",
        lambda path: [
            SimpleNamespace(
                run_key="exact_atlas_full_setup::2026-04-07T18:21:12.252546+00:00",
                scenario_id="exact_atlas_full_setup",
                review_status="approved",
                human_label="good",
                expected_event="Atlas Mountain Race",
                expected_component="full_setup",
                corrected_event="Atlas Mountain Race",
                corrected_component="full_setup",
                corrected_policy_mode="strict_grounded",
                review_notes="Keep exact grounding honest.",
                review_timestamp="2026-04-07T18:21:12.252546+00:00",
            )
        ],
    )
    monkeypatch.setattr(
        mod,
        "find_relevant_reviews",
        lambda reviews, **kwargs: reviews,
    )
    monkeypatch.setattr(
        mod,
        "format_review_context",
        lambda reviews: "- scenario_id=exact_atlas_full_setup; review_status=approved; human_label=good; corrected_policy_mode=strict_grounded",
    )

    rec, trace = mod.recommend_setup_with_trace("Give me tyres for Pirenaica")

    assert isinstance(rec, SetupRecommendation)
    assert prompt_holder["prompt"] is not None
    assert "Human review hints:" in prompt_holder["prompt"]
    assert "corrected_policy_mode=strict_grounded" in prompt_holder["prompt"]
