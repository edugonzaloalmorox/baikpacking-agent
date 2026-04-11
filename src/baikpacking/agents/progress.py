"""Structured progress events for the recommendation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ProgressStageKey = Literal[
    "resolving_event",
    "classifying_intent",
    "searching_riders",
    "selecting_policy",
    "writing_recommendation",
]


class RecommendationProgress(BaseModel):
    """One stage update emitted while a recommendation is running."""

    model_config = ConfigDict(extra="ignore")

    stage_key: ProgressStageKey
    stage_label: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = Field(default_factory=dict)


ProgressCallback = Callable[[RecommendationProgress], None]


def build_progress_event(
    stage_key: ProgressStageKey,
    stage_label: str,
    *,
    details: Optional[dict[str, Any]] = None,
) -> RecommendationProgress:
    """Create a normalized progress event."""
    return RecommendationProgress(
        stage_key=stage_key,
        stage_label=stage_label,
        details=details or {},
    )


def emit_progress(
    callback: ProgressCallback | None,
    stage_key: ProgressStageKey,
    stage_label: str,
    *,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a progress event when a callback is available."""
    if callback is None:
        return
    try:
        callback(build_progress_event(stage_key, stage_label, details=details))
    except Exception:
        pass
