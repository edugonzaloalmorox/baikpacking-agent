from __future__ import annotations

import asyncio
import json
from pathlib import Path

from baikpacking.agents.live_feedback import (
    append_live_feedback,
    build_live_feedback_record,
    join_live_runs_with_feedback,
    load_live_feedback,
)
from baikpacking.api.schemas import FeedbackRequest
from baikpacking.api.service import RecommendationService


def test_feedback_persistence_roundtrip(tmp_path: Path, monkeypatch):
    import baikpacking.agents.live_feedback as feedback_mod

    monkeypatch.setattr(feedback_mod, "DEFAULT_LIVE_FEEDBACK_PATH", tmp_path / "live_feedback.jsonl")

    service = RecommendationService()
    response = asyncio.run(
        service.submit_feedback(
            FeedbackRequest(
                run_id="run-123",
                feedback="thumbs_down",
                comment="It matched the wrong event",
            ),
            request_meta={"method": "POST", "path": "/feedback", "user_agent": "pytest"},
        )
    )

    assert response.run_id == "run-123"
    assert response.feedback == "thumbs_down"

    feedback_path = tmp_path / "live_feedback.jsonl"
    assert feedback_path.exists()
    rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-123"
    assert row["feedback"] == "thumbs_down"
    assert row["comment"] == "It matched the wrong event"
    assert row["request_meta"]["path"] == "/feedback"


def test_join_live_runs_with_feedback(tmp_path: Path):
    feedback_path = tmp_path / "live_feedback.jsonl"
    run_id = "run-abc"

    append_live_feedback(
        build_live_feedback_record(
            run_id=run_id,
            feedback="thumbs_up",
            comment="",
            request_meta={"path": "/feedback"},
            timestamp="2026-04-10T00:00:00+00:00",
        ),
        feedback_path,
    )
    append_live_feedback(
        build_live_feedback_record(
            run_id=run_id,
            feedback="thumbs_down",
            comment="Actually the event was wrong",
            request_meta={"path": "/feedback"},
            timestamp="2026-04-10T00:05:00+00:00",
        ),
        feedback_path,
    )

    feedback_records = load_live_feedback(feedback_path)
    joined = join_live_runs_with_feedback(
        [{"run_id": run_id, "query": "What tyres should I use?"}],
        feedback_records,
    )

    assert joined[0]["feedback_count"] == 2
    assert joined[0]["latest_feedback"]["feedback"] == "thumbs_down"
    assert joined[0]["feedback_events"][0]["run_id"] == run_id
