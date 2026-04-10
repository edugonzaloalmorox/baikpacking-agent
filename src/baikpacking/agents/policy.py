from baikpacking.agents.orchestration_models import (
    EvidenceSummary,
    EventResolutionResult,
    RecommendationPolicy,
)


def select_policy(
    *,
    event_match_type: str,
    matched_event_name: str | None,
    retrieval_source: str,
    exact_event_hit_count: int,
    evidence_strength: str,
    allow_exact_grounding: bool = True,
) -> RecommendationPolicy:
    is_exact_event_retrieval = retrieval_source == "exact_event" and exact_event_hit_count > 0
    is_similar_event_retrieval = retrieval_source == "similar_event"
    is_unknown_retrieval = retrieval_source == "unknown_global"

    if is_exact_event_retrieval and allow_exact_grounding and evidence_strength == "strong":
        return RecommendationPolicy(
            mode="strict_grounded",
            allow_specific_brands=True,
            allow_specific_specs=True,
            allow_event_specific_claims=True,
            notes=[
                "exact_event_retrieval",
                f"matched_event={matched_event_name}" if matched_event_name else "matched_event=unknown",
                f"event_match_type={event_match_type}",
                "strong_evidence",
            ],
        )

    if is_exact_event_retrieval and evidence_strength in {"moderate", "strong"}:
        return RecommendationPolicy(
            mode="pattern_based",
            allow_specific_brands=allow_exact_grounding,
            allow_specific_specs=True,
            allow_event_specific_claims=allow_exact_grounding,
            notes=[
                "exact_event_retrieval",
                f"matched_event={matched_event_name}" if matched_event_name else "matched_event=unknown",
                f"event_match_type={event_match_type}",
                f"{evidence_strength}_evidence",
                "exact_grounding_disabled" if not allow_exact_grounding else "exact_grounding_allowed",
            ],
        )

    if is_similar_event_retrieval and evidence_strength in {"moderate", "strong"}:
        return RecommendationPolicy(
            mode="pattern_based",
            allow_specific_brands=False,
            allow_specific_specs=True,
            allow_event_specific_claims=False,
            notes=[
                "similar_event_retrieval",
                f"matched_event={matched_event_name}" if matched_event_name else "matched_event=unknown",
                f"event_match_type={event_match_type}",
                f"{evidence_strength}_evidence",
            ],
        )

    return RecommendationPolicy(
        mode="generic_fallback",
        allow_specific_brands=False,
        allow_specific_specs=False,
        allow_event_specific_claims=False,
        notes=[
            "unknown_event" if is_unknown_retrieval else retrieval_source,
            f"matched_event={matched_event_name}" if matched_event_name else "matched_event=unknown",
            f"event_match_type={event_match_type}",
            f"{evidence_strength}_evidence",
        ],
    )


def choose_recommendation_policy(
    event_resolution: EventResolutionResult,
    evidence_summary: EvidenceSummary,
) -> RecommendationPolicy:
    return select_policy(
        event_match_type=event_resolution.match_type,
        matched_event_name=event_resolution.canonical_name or event_resolution.display_name,
        retrieval_source="exact_event" if event_resolution.is_trusted_exact else "unknown_global",
        exact_event_hit_count=1 if event_resolution.is_trusted_exact else 0,
        evidence_strength=evidence_summary.evidence_strength,
        allow_exact_grounding=event_resolution.is_trusted_exact,
    )
