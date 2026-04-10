from baikpacking.agents.event_context_resolution import _build_descriptor_query
from baikpacking.agents.models import QueryIntent
from baikpacking.agents.orchestration_models import EventContextSummary, EventResolutionResult, RetrievalPlan
from baikpacking.agents.query_intent import _build_retrieval_intent_bundle


def build_retrieval_plan(
    event_resolution: EventResolutionResult,
    event_context_summary: EventContextSummary,
    intent: QueryIntent,
    user_query: str,
    *,
    allow_exact_grounding: bool = True,
) -> RetrievalPlan:
    if event_resolution.canonical_name and _normalized_match_type(event_resolution) in {"exact", "alias", "trusted_exact"}:
        event_name_for_retrieval = event_resolution.canonical_name
    else:
        event_name_for_retrieval = event_context_summary.event_family or event_resolution.display_name

    descriptor = _build_descriptor_query(
        event_name=event_resolution.display_name,
        event_context=event_context_summary.web_context_text,
        user_question=user_query,
    )
    intent_bundle = _build_retrieval_intent_bundle(
        descriptor=descriptor,
        intent=intent,
    )

    if intent.component != "full_setup" and intent_bundle.component_query:
        primary_query = intent_bundle.component_query
        fallback_query = intent_bundle.broad_query
        fallback_reasoning = "component_specific_then_broaden"
    else:
        primary_query = intent_bundle.broad_query
        fallback_query = intent_bundle.component_query
        fallback_reasoning = "broad_then_component_context"

    return RetrievalPlan(
        query_component=intent.component,
        use_exact_event=bool(allow_exact_grounding and event_resolution.is_trusted_exact),
        event_name_for_retrieval=event_name_for_retrieval,
        descriptor_query=descriptor["descriptor_query"],
        descriptor_query_with_intent=descriptor["descriptor_query_with_intent"],
        primary_query=primary_query,
        fallback_query=fallback_query,
        fallback_reasoning=fallback_reasoning,
        intent_bundle=intent_bundle,
    )


def _normalized_match_type(event_resolution: EventResolutionResult) -> str:
    return str(event_resolution.match_type).strip().lower()
