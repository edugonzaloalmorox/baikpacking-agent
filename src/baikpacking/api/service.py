"""Service layer for the bikepacking HTTP API."""



import logging
from typing import Any, Optional

import anyio

from baikpacking.agents.models import QueryIntent
from baikpacking.agents.orchestration_models import (
    EvidenceSummary,
    EventResolutionResult,
    RecommendationPolicy,
    RetrievalPlan,
)
from baikpacking.agents.recommender_agent import recommend_setup_with_trace
from baikpacking.api.schemas import (
    HealthResponse,
    RecommendationDebug,
    RecommendRequest,
    RecommendResponse,
    ReadyResponse,
    TraceEventSchema,
)
from baikpacking.db.db_connection import ping_db

logger = logging.getLogger(__name__)


class ApiServiceError(RuntimeError):
    """Domain error raised by the API service layer."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _trace_calls(trace: Any) -> list[dict[str, Any]]:
    if trace is None:
        return []
    calls = getattr(trace, "calls", None)
    if isinstance(calls, list):
        return calls
    entries = getattr(trace, "entries", None)
    if isinstance(entries, list):
        return entries
    if isinstance(trace, list):
        return trace
    return []


def _find_trace_step(trace: Any, tool_name: str) -> Optional[dict[str, Any]]:
    for step in reversed(_trace_calls(trace)):
        if isinstance(step, dict) and step.get("tool") == tool_name:
            return step
    return None


def _model_from_trace(
    trace: Any,
    tool_name: str,
    model_cls: type[Any],
    fallback: Any,
) -> Any:
    step = _find_trace_step(trace, tool_name)
    if not step:
        return fallback
    result = step.get("result")
    if not isinstance(result, dict):
        return fallback
    try:
        return model_cls.model_validate(result)
    except Exception:
        return fallback


def _build_debug(trace: Any) -> RecommendationDebug:
    retrieval_plan = _model_from_trace(trace, "retrieval_plan", RetrievalPlan, None)
    trace_events = [TraceEventSchema.model_validate(step) for step in _trace_calls(trace)]
    return RecommendationDebug(retrieval_plan=retrieval_plan, trace=trace_events)


def _default_event_resolution(query: str, recommendation_event: Optional[str]) -> EventResolutionResult:
    resolved = recommendation_event or query or "Unknown event"
    return EventResolutionResult(
        raw_query_event=query or None,
        canonical_name=recommendation_event or None,
        display_name=resolved,
        match_type="unknown",
        confidence=0.0,
        is_trusted_exact=False,
    )


def _default_intent() -> QueryIntent:
    return QueryIntent()


def _default_evidence() -> EvidenceSummary:
    return EvidenceSummary()


def _default_policy() -> RecommendationPolicy:
    return RecommendationPolicy()


class RecommendationService:
    """Small service wrapper around the existing recommendation pipeline."""

    async def health(self) -> HealthResponse:
        return HealthResponse()

    async def ready(self) -> ReadyResponse:
        try:
            database = await anyio.to_thread.run_sync(ping_db)
        except Exception as exc:
            logger.exception("readiness check failed")
            raise ApiServiceError(f"database not ready: {exc}", status_code=503) from exc
        return ReadyResponse(database=database)

    async def recommend(self, request: RecommendRequest) -> RecommendResponse:
        query = request.query.strip()
        if not query:
            raise ApiServiceError("query must not be empty", status_code=400)

        logger.info("recommend_request query=%r include_debug=%s", query, request.include_debug)

        try:
            recommendation, trace = await anyio.to_thread.run_sync(recommend_setup_with_trace, query)
        except Exception as exc:
            logger.exception("recommendation failed for query=%r", query)
            raise ApiServiceError(f"recommendation failed: {exc}", status_code=503) from exc

        event_resolution = _model_from_trace(
            trace,
            "event_resolution",
            EventResolutionResult,
            _default_event_resolution(query, recommendation.event),
        )
        intent = _model_from_trace(trace, "intent_classification", QueryIntent, _default_intent())
        evidence = _model_from_trace(trace, "evidence_summary", EvidenceSummary, _default_evidence())
        policy = _model_from_trace(trace, "policy_selection", RecommendationPolicy, _default_policy())

        debug = _build_debug(trace) if request.include_debug else None

        return RecommendResponse(
            query=query,
            resolved_event=event_resolution,
            intent=intent,
            recommendation=recommendation,
            evidence=evidence,
            policy=policy,
            debug=debug,
        )


_SERVICE = RecommendationService()


def get_recommendation_service() -> RecommendationService:
    """FastAPI dependency that returns the singleton API service."""
    return _SERVICE
