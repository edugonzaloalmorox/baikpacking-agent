"""Append-only storage for user feedback on live recommendations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_LIVE_FEEDBACK_PATH = Path(__file__).resolve().parents[3] / "data/eval/live_feedback.jsonl"
FeedbackValue = Literal["thumbs_up", "thumbs_down"]


class LiveFeedbackRecord(BaseModel):
    """Persisted user feedback for a single live recommendation run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feedback: FeedbackValue
    comment: str = ""
    request_meta: dict[str, Any] = Field(default_factory=dict)


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _safe_json(dump())
        except Exception:
            return None
    return None


def _normalize_comment(comment: Any) -> str:
    if comment is None:
        return ""
    text = " ".join(str(comment).split()).strip()
    return text


def build_live_feedback_record(
    *,
    run_id: str,
    feedback: FeedbackValue,
    comment: str | None = None,
    request_meta: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> LiveFeedbackRecord:
    """Build a deterministic feedback record suitable for JSONL storage."""
    return LiveFeedbackRecord(
        run_id=run_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        feedback=feedback,
        comment=_normalize_comment(comment),
        request_meta=_safe_json(dict(request_meta or {})) or {},
    )


def append_live_feedback(record: LiveFeedbackRecord, path: str | Path | None = None) -> None:
    """Append one feedback record to the JSONL sink."""
    feedback_path = Path(path or DEFAULT_LIVE_FEEDBACK_PATH)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")


def load_live_feedback(path: str | Path) -> list[LiveFeedbackRecord]:
    """Load persisted live feedback records from JSONL."""
    feedback_path = Path(path)
    if not feedback_path.exists():
        return []

    records: list[LiveFeedbackRecord] = []
    with feedback_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(LiveFeedbackRecord.model_validate_json(line))
            except Exception:
                continue
    return records


def latest_feedback_by_run_id(records: list[LiveFeedbackRecord]) -> dict[str, LiveFeedbackRecord]:
    """Return the newest feedback entry for each live run."""
    latest: dict[str, LiveFeedbackRecord] = {}
    for record in records:
        current = latest.get(record.run_id)
        if current is None or record.timestamp >= current.timestamp:
            latest[record.run_id] = record
    return latest


def join_live_runs_with_feedback(
    runs: list[Mapping[str, Any]],
    feedback_records: list[LiveFeedbackRecord],
) -> list[dict[str, Any]]:
    """Attach feedback summaries to live runs for downstream analysis."""
    grouped: dict[str, list[LiveFeedbackRecord]] = {}
    for record in feedback_records:
        grouped.setdefault(record.run_id, []).append(record)

    joined: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run.get("run_id") or "")
        records = grouped.get(run_id, [])
        latest = None
        if records:
            latest = sorted(records, key=lambda item: item.timestamp)[-1]
        merged = dict(run)
        merged["feedback_events"] = [item.model_dump(mode="json") for item in records]
        merged["feedback_count"] = len(records)
        merged["latest_feedback"] = latest.model_dump(mode="json") if latest else None
        joined.append(merged)
    return joined
