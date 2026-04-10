"""Pydantic schemas for the bikepacking HTTP API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from baikpacking.agents.models import QueryIntent, SetupRecommendation
from baikpacking.agents.guardrails import GuardDecision
from baikpacking.agents.orchestration_models import (
    EvidenceSummary,
    EventResolutionResult,
    RecommendationPolicy,
    RetrievalPlan,
)


class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, description="User question or recommendation request.")
    include_debug: bool = Field(
        default=False,
        description="Include the raw trace and retrieval-plan debug payload when true.",
    )


class TraceEventSchema(BaseModel):
    """Serializable trace event exposed by the API when debug is requested."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    elapsed_ms: float = 0.0


class RecommendationDebug(BaseModel):
    """Optional debug payload for recommendation requests."""

    retrieval_plan: Optional[RetrievalPlan] = None
    trace: list[TraceEventSchema] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    """Structured recommendation response."""

    query: str
    status: Literal["success", "skipped"] = "success"
    message: Optional[str] = None
    guard: Optional[GuardDecision] = None
    resolved_event: Optional[EventResolutionResult] = None
    intent: Optional[QueryIntent] = None
    recommendation: Optional[SetupRecommendation] = None
    evidence: Optional[EvidenceSummary] = None
    policy: Optional[RecommendationPolicy] = None
    debug: Optional[RecommendationDebug] = None


class ReadyResponse(BaseModel):
    """Readiness response payload."""

    status: str = "ready"
    database: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Liveness response payload."""

    status: str = "ok"


class ErrorResponse(BaseModel):
    """Structured error payload returned by API exception handlers."""

    error: str
    detail: str
    request_id: Optional[str] = None
