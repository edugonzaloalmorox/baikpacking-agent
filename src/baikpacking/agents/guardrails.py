"""Deterministic guardrails for deciding whether to recommend at all."""


import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from baikpacking.agents.models import QueryIntent
from baikpacking.agents.orchestration_models import EventResolutionResult
from baikpacking.tools.call_trace import CallTrace


_BIKEPACKING_KEYWORDS = [
    "bikepacking",
    "bike pack",
    "bikepack",
    "bike",
    "bikes",
    "bicycle",
    "bicycles",
    "setup",
    "gear",
    "tyre",
    "tyres",
    "tire",
    "tires",
    "bag",
    "bags",
    "drivetrain",
    "wheel",
    "wheels",
    "sleep",
    "navigation",
    "race",
    "event",
    "ultra",
]
_RECOMMENDATION_PATTERNS = [
    r"\brecommend\b",
    r"\bsuggest\b",
    r"\bwhat should i use\b",
    r"\bwhat do you recommend\b",
    r"\bwhich .* should i use\b",
    r"\bbest setup\b",
    r"\bsetup for\b",
    r"\bgear for\b",
]
_STRONG_EVENT_MATCH_TYPES = {"exact", "alias", "trusted_exact"}
_WEAK_EVENT_MATCH_TYPES = {"weak_candidate", "unknown"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _query_mentions_bikepacking(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(keyword in normalized for keyword in _BIKEPACKING_KEYWORDS)


def _query_explicitly_asks_for_recommendation(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(re.search(pattern, normalized) for pattern in _RECOMMENDATION_PATTERNS)


def _has_component_terms(intent: QueryIntent) -> bool:
    return bool([term for term in intent.component_terms if _normalize_text(term)])


def _strong_event_match(event_resolution: EventResolutionResult) -> bool:
    return bool(
        event_resolution.is_trusted_exact
        and _normalize_text(event_resolution.match_type) in _STRONG_EVENT_MATCH_TYPES
    )


def _weak_or_untrusted_event(event_resolution: EventResolutionResult) -> bool:
    return (
        _normalize_text(event_resolution.match_type) in _WEAK_EVENT_MATCH_TYPES
        or not event_resolution.is_trusted_exact
    )


class GuardDecision(BaseModel):
    """Deterministic recommendation guard decision."""

    model_config = ConfigDict(extra="ignore")

    allow_recommendation: bool = True
    reason: str = ""
    guard_type: str = ""
    allow_exact_grounding: bool = True
    user_message: str = ""


class RecommendationGuardBlocked(RuntimeError):
    """Raised when a query must not proceed to retrieval or recommendation."""

    def __init__(self, decision: GuardDecision, *, trace: CallTrace):
        super().__init__(decision.reason or decision.guard_type or "recommendation blocked")
        self.decision = decision
        self.trace = trace


def should_recommend(
    query: str,
    event_resolution: EventResolutionResult,
    intent: QueryIntent,
) -> GuardDecision:
    """Return a deterministic decision about whether to recommend at all."""
    strong_event = _strong_event_match(event_resolution)
    weak_event = _weak_or_untrusted_event(event_resolution)
    mentions_bikepacking = _query_mentions_bikepacking(query)
    explicit_request = _query_explicitly_asks_for_recommendation(query)
    has_component_terms = _has_component_terms(intent)
    low_intent = intent.confidence < 0.5
    generic_intent = intent.component == "full_setup" or not has_component_terms

    if not mentions_bikepacking and not strong_event:
        return GuardDecision(
            allow_recommendation=False,
            reason="Query does not appear to be about bikepacking or a known event.",
            guard_type="out_of_domain",
            allow_exact_grounding=False,
            user_message="This query does not appear to be a bikepacking event or setup request.",
        )

    if weak_event and (low_intent or generic_intent):
        return GuardDecision(
            allow_recommendation=False,
            reason="Weak or untrusted event match combined with low-confidence or generic intent.",
            guard_type="combined_safety",
            allow_exact_grounding=False,
            user_message="Please specify an event or gear-related question.",
        )

    if low_intent and not has_component_terms and not explicit_request:
        return GuardDecision(
            allow_recommendation=False,
            reason="Intent confidence is too low and the query does not explicitly ask for a recommendation.",
            guard_type="low_intent_confidence",
            allow_exact_grounding=False,
            user_message="Please specify an event or gear-related question.",
        )

    if weak_event:
        return GuardDecision(
            allow_recommendation=True,
            reason="Event match is weak or untrusted; exact grounding is disabled.",
            guard_type="weak_event",
            allow_exact_grounding=False,
            user_message="I can help, but I will avoid exact-event grounding and keep the answer cautious.",
        )

    return GuardDecision(
        allow_recommendation=True,
        reason="Query is in domain and event signal is strong enough to continue.",
        guard_type="allow",
        allow_exact_grounding=True,
        user_message="",
    )


def guard_trace_payload(decision: GuardDecision, query: str) -> dict[str, Any]:
    """Return a small JSON-serializable payload for tracing the guard decision."""
    return {
        "guard_id": uuid.uuid4().hex,
        "query": query,
        **decision.model_dump(),
    }
