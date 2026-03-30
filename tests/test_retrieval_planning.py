from baikpacking.agents.models import QueryIntent
from baikpacking.agents.orchestration_models import EventContextSummary, EventResolutionResult
from baikpacking.agents.retrieval_planning import build_retrieval_plan


def test_build_retrieval_plan_exact_event_full_setup():
    event_resolution = EventResolutionResult(
        display_name="Atlas Mountain Race",
        canonical_name="Atlas Mountain Race",
        match_type="exact",
        confidence=0.98,
        is_trusted_exact=True,
    )
    event_context_summary = EventContextSummary(
        requested_event_name="Atlas Mountain Race",
        web_context_text="Remote mountain ultra",
        archetype="mountain_offroad_ultra",
        surface_family="mixed_offroad",
    )
    intent = QueryIntent(component="full_setup", confidence=0.25, component_terms=[])

    plan = build_retrieval_plan(
        event_resolution=event_resolution,
        event_context_summary=event_context_summary,
        intent=intent,
        user_query="Recommend a setup for Atlas Mountain Race",
    )

    assert plan.use_exact_event is True
    assert plan.event_name_for_retrieval == "Atlas Mountain Race"
    assert plan.primary_query == plan.descriptor_query
    assert plan.fallback_query == plan.descriptor_query_with_intent


def test_build_retrieval_plan_low_confidence_event_represents_inferred_context():
    event_resolution = EventResolutionResult(
        display_name="North Cape Tarifa",
        canonical_name=None,
        match_type="unknown",
        confidence=0.15,
        is_trusted_exact=False,
    )
    event_context_summary = EventContextSummary(
        requested_event_name="North Cape Tarifa",
        web_context_text="Ultra-distance Europe crossing race",
        event_family="trans-europe ultra",
        family_confidence=0.35,
        archetype="road_ultra",
        surface_family="road",
    )
    intent = QueryIntent(component="full_setup", confidence=0.25, component_terms=[])

    plan = build_retrieval_plan(
        event_resolution=event_resolution,
        event_context_summary=event_context_summary,
        intent=intent,
        user_query="North Cape Tarifa race",
    )

    assert plan.use_exact_event is False
    assert plan.event_name_for_retrieval == "trans-europe ultra"
    assert "North Cape Tarifa" in plan.descriptor_query
    assert plan.primary_query == plan.descriptor_query


def test_build_retrieval_plan_component_query_matches_previous_shape():
    event_resolution = EventResolutionResult(
        display_name="Transiberica",
        canonical_name="Transiberica",
        match_type="exact",
        confidence=0.98,
        is_trusted_exact=True,
    )
    event_context_summary = EventContextSummary(
        requested_event_name="Transiberica",
        web_context_text="Road ultra race across Spain",
        archetype="road_ultra",
        surface_family="road",
    )
    intent = QueryIntent(component="bags", confidence=0.5, component_terms=["bag", "bags"])

    plan = build_retrieval_plan(
        event_resolution=event_resolution,
        event_context_summary=event_context_summary,
        intent=intent,
        user_query="What bags should I use for Transiberica?",
    )

    assert plan.primary_query is not None
    assert "Focus: bikepacking bags" in plan.primary_query
    assert plan.fallback_query == plan.descriptor_query
