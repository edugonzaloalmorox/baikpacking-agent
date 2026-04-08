"""Utilities for storing and reusing lightweight human eval feedback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_REVIEWS_PATH = Path(__file__).resolve().parents[3] / "data/eval/scenario_reviews.jsonl"
ReviewStatus = Literal["pending", "approved", "rejected", "needs_followup"]
HumanLabel = Literal["good", "bad", "partially_good"]


class ScenarioReview(BaseModel):
    """Persisted reviewer feedback for a single eval run."""

    model_config = ConfigDict(extra="ignore")

    run_key: str
    scenario_id: str
    run_timestamp: Optional[str] = None
    expected_event: Optional[str] = None
    expected_component: Optional[str] = None
    expected_policy_mode: Optional[str] = None
    review_status: ReviewStatus = "pending"
    human_label: HumanLabel = "partially_good"
    corrected_event: Optional[str] = None
    corrected_component: Optional[str] = None
    corrected_policy_mode: Optional[str] = None
    review_notes: Optional[str] = None
    review_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split()).strip()
    if not text or text.lower() in {"<na>", "nan", "none"}:
        return ""
    return text.lower()


def _string_or_none(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    return text or None


def _first_text(*values: Any) -> str:
    for value in values:
        text = _normalize_text(value)
        if text:
            return text
    return ""


def build_run_key(run: Mapping[str, Any]) -> str:
    """Build a stable key for a scenario run."""
    scenario_id = _first_text(run.get("scenario_id"))
    timestamp = _first_text(run.get("timestamp"), run.get("run_timestamp"))
    if scenario_id and timestamp:
        return f"{scenario_id}::{timestamp}"
    if scenario_id:
        return scenario_id
    return _normalize_text(run.get("resolved_event_name") or run.get("event_name") or "unknown")


def review_from_run(run: Mapping[str, Any], existing: Optional[ScenarioReview] = None) -> ScenarioReview:
    """Create a review object seeded from a scenario run and optional prior review."""
    run_key = build_run_key(run)
    data: dict[str, Any] = {
        "run_key": run_key,
        "scenario_id": _string_or_none(run.get("scenario_id")) or "",
        "run_timestamp": _string_or_none(run.get("timestamp")) or _string_or_none(run.get("run_timestamp")),
        "expected_event": _string_or_none(run.get("expected_event")),
        "expected_component": _string_or_none(run.get("expected_component")),
        "expected_policy_mode": _string_or_none(run.get("expected_policy_mode")),
    }
    if existing is not None:
        data.update(existing.model_dump())
        data["run_key"] = run_key
        data["scenario_id"] = _string_or_none(run.get("scenario_id")) or existing.scenario_id or ""
        data["run_timestamp"] = (
            _string_or_none(run.get("timestamp")) or _string_or_none(run.get("run_timestamp"))
            or existing.run_timestamp
        )
        data["expected_event"] = _string_or_none(run.get("expected_event")) or existing.expected_event
        data["expected_component"] = _string_or_none(run.get("expected_component")) or existing.expected_component
        data["expected_policy_mode"] = (
            _string_or_none(run.get("expected_policy_mode")) or existing.expected_policy_mode
        )
    return ScenarioReview(**data)


def load_reviews(path: str | Path) -> list[ScenarioReview]:
    """Load persisted reviews from JSONL."""
    review_path = Path(path)
    if not review_path.exists():
        return []

    reviews: list[ScenarioReview] = []
    with review_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                reviews.append(ScenarioReview.model_validate_json(line))
            except Exception:
                continue
    return reviews


def save_review(review: ScenarioReview, path: str | Path) -> None:
    """Upsert a review by run_key and persist JSONL."""
    review_path = Path(path)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(review_path)
    by_key = {item.run_key: item for item in reviews}
    by_key[review.run_key] = review

    ordered = sorted(by_key.values(), key=lambda item: (item.review_timestamp, item.run_key))
    with review_path.open("w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(item.model_dump_json())
            handle.write("\n")


def _event_matches(review: ScenarioReview, normalized_event: str) -> bool:
    candidates = [
        review.expected_event,
        review.corrected_event,
    ]
    return any(_normalize_text(candidate) == normalized_event for candidate in candidates if candidate)


def _component_matches(review: ScenarioReview, normalized_component: str) -> bool:
    candidates = [
        review.expected_component,
        review.corrected_component,
    ]
    return any(_normalize_text(candidate) == normalized_component for candidate in candidates if candidate)


def find_relevant_reviews(
    reviews: Iterable[ScenarioReview],
    *,
    expected_event: Optional[str] = None,
    expected_component: Optional[str] = None,
    scenario_id: Optional[str] = None,
    limit: int = 3,
) -> list[ScenarioReview]:
    """Return a small set of feedback examples relevant to the current query."""
    normalized_event = _normalize_text(expected_event)
    normalized_component = _normalize_text(expected_component)
    normalized_scenario_id = _normalize_text(scenario_id)

    scored: list[tuple[int, str, ScenarioReview]] = []
    for review in reviews:
        if review.review_status in {"pending", "rejected"}:
            continue

        score = 0
        if normalized_scenario_id and _normalize_text(review.scenario_id) == normalized_scenario_id:
            score += 100
        if normalized_event and _event_matches(review, normalized_event):
            score += 40
        if normalized_component and _component_matches(review, normalized_component):
            score += 20
        if review.review_status in {"approved", "needs_followup"}:
            score += 5
        if review.human_label in {"good", "partially_good"}:
            score += 3
        if score <= 0:
            continue

        scored.append((score, review.review_timestamp, review))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[: max(0, int(limit))]]


def format_review_context(reviews: Iterable[ScenarioReview]) -> str:
    """Render a compact prompt-safe review summary."""
    lines: list[str] = []
    for review in reviews:
        parts = [
            f"scenario_id={review.scenario_id}",
            f"review_status={review.review_status}",
            f"human_label={review.human_label}",
        ]
        if review.expected_event:
            parts.append(f"expected_event={review.expected_event}")
        if review.expected_component:
            parts.append(f"expected_component={review.expected_component}")
        if review.corrected_event:
            parts.append(f"corrected_event={review.corrected_event}")
        if review.corrected_component:
            parts.append(f"corrected_component={review.corrected_component}")
        if review.corrected_policy_mode:
            parts.append(f"corrected_policy_mode={review.corrected_policy_mode}")
        if review.review_notes:
            notes = " ".join(review.review_notes.split())
            parts.append(f"notes={notes[:220]}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)
