"""Service layer for the bikepacking HTTP API."""

from __future__ import annotations

from functools import partial
import logging
import time
import uuid
from typing import Any, Mapping, Optional

from baikpacking.agents.models import QueryIntent
from baikpacking.agents.guardrails import RecommendationGuardBlocked
from baikpacking.agents.live_evaluation import append_live_run, build_live_run_record
from baikpacking.agents.live_feedback import append_live_feedback, build_live_feedback_record
from baikpacking.agents.progress import ProgressCallback
from baikpacking.agents.orchestration_models import (
    EvidenceSummary,
    EventResolutionResult,
    RecommendationPolicy,
    RetrievalPlan,
)
from baikpacking.agents.recommender_agent import recommend_setup_with_trace
from baikpacking.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
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

    async def recommend(
        self,
        request: RecommendRequest,
        *,
        request_meta: Mapping[str, Any] | None = None,
    ) -> RecommendResponse:
        import anyio

        return await anyio.to_thread.run_sync(partial(self.recommend_sync, request, request_meta=request_meta))

    def recommend_sync(
        self,
        request: RecommendRequest,
        *,
        request_meta: Mapping[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RecommendResponse:
        query = request.query.strip()
        status = "success"
        error_text: str | None = None
        response_payload: dict[str, Any] | None = None
        trace: Any = None
        run_id = uuid.uuid4().hex
        started = time.perf_counter()

        try:
            if not query:
                status = "failure"
                error_text = "query must not be empty"
                raise ApiServiceError(error_text, status_code=400)

            logger.info("recommend_request query=%r include_debug=%s", query, request.include_debug)

            try:
                try:
                    recommendation, trace = recommend_setup_with_trace(query, progress_callback=progress_callback)
                except TypeError as exc:
                    if "unexpected keyword argument 'progress_callback'" not in str(exc):
                        raise
                    recommendation, trace = recommend_setup_with_trace(query)
            except RecommendationGuardBlocked as exc:
                logger.info(
                    "recommendation skipped for query=%r guard_type=%s",
                    query,
                    exc.decision.guard_type,
                )
                status = "skipped"
                trace = exc.trace
                response = RecommendResponse(
                    run_id=run_id,
                    query=query,
                    status="skipped",
                    message=exc.decision.user_message or exc.decision.reason,
                    guard=exc.decision,
                )
                response_payload = response.model_dump(mode="json")
                return response
            except Exception as exc:
                logger.exception("recommendation failed for query=%r", query)
                status = "failure"
                error_text = f"recommendation failed: {exc}"
                raise ApiServiceError(error_text, status_code=503) from exc

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
            response = RecommendResponse(
                run_id=run_id,
                query=query,
                resolved_event=event_resolution,
                intent=intent,
                recommendation=recommendation,
                evidence=evidence,
                policy=policy,
                debug=debug,
            )

            response_payload = response.model_dump(mode="json")
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._write_live_eval(
                run_id=run_id,
                query=query,
                status=status,
                error_text=error_text,
                response_payload=response_payload,
                trace=trace,
                latency_ms=latency_ms,
                request_meta=request_meta,
            )

    def _write_live_eval(
        self,
        *,
        run_id: str,
        query: str,
        status: str,
        error_text: str | None,
        response_payload: dict[str, Any] | None,
        trace: Any,
        latency_ms: float,
        request_meta: Mapping[str, Any] | None = None,
    ) -> None:
        record = build_live_run_record(
            run_id=run_id,
            query=query,
            status=status,
            error=error_text,
            response=response_payload,
            trace=trace,
            latency_ms=latency_ms,
            request_meta=request_meta,
        )
        try:
            append_live_run(record)
        except Exception:
            logger.exception("failed to persist live eval row")

    async def submit_feedback(
        self,
        request: FeedbackRequest,
        *,
        request_meta: Mapping[str, Any] | None = None,
    ) -> FeedbackResponse:
        import anyio

        return await anyio.to_thread.run_sync(partial(self.submit_feedback_sync, request, request_meta=request_meta))

    def submit_feedback_sync(
        self,
        request: FeedbackRequest,
        *,
        request_meta: Mapping[str, Any] | None = None,
    ) -> FeedbackResponse:
        run_id = request.run_id.strip()
        if not run_id:
            raise ApiServiceError("run_id must not be empty", status_code=400)

        record = build_live_feedback_record(
            run_id=run_id,
            feedback=request.feedback,
            comment=request.comment,
            request_meta=request_meta,
        )
        try:
            append_live_feedback(record)
        except Exception:
            logger.exception("failed to persist live feedback row")
            raise ApiServiceError("failed to persist feedback", status_code=503) from None

        return FeedbackResponse(
            run_id=record.run_id,
            feedback=record.feedback,
            timestamp=record.timestamp,
        )


_SERVICE = RecommendationService()


def get_recommendation_service() -> RecommendationService:
    """FastAPI dependency that returns the singleton API service."""
    return _SERVICE
