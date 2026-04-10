from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baikpacking.agents.live_evaluation import build_live_run_record, classify_failure_kind
from baikpacking.api.service import RecommendationService
from baikpacking.agents.models import SetupCore, SetupRecommendation


def _fake_trace() -> SimpleNamespace:
    return SimpleNamespace(
        calls=[
            {
                "tool": "policy_selection",
                "args": {
                    "event_match_type": "exact",
                    "matched_event_name": "Atlas Mountain Race",
                    "retrieval_source": "exact_event",
                    "exact_event_hit_count": 2,
                },
                "result": {
                    "mode": "strict_grounded",
                    "allow_specific_brands": True,
                    "allow_specific_specs": True,
                    "allow_event_specific_claims": True,
                    "notes": ["exact_event_retrieval"],
                },
                "elapsed_ms": 0.0,
            }
        ]
    )


def test_build_live_run_record_captures_quality_signals():
    response = {
        "query": "What tyres do you recommend for Atlas Mountain Race?",
        "resolved_event": {
            "display_name": "Atlas Mountain Race",
            "canonical_name": "Atlas Mountain Race",
            "match_type": "exact",
            "is_trusted_exact": True,
        },
        "intent": {"component": "tyres", "confidence": 0.92},
        "recommendation": {
            "event": "Atlas Mountain Race",
            "summary": "Use a grounded tyre setup.",
            "reasoning": "Exact event evidence supports this.",
            "recommended_setup": {
                "bike_type": "gravel bike",
                "wheels": "700c",
                "tyres": "45mm semi-slick",
                "drivetrain": "GRX 1x",
                "bags": "frame bag",
                "sleep_system": "bivy",
            },
        },
        "evidence": {
            "rider_count": 3,
            "component_hit_count": 2,
            "evidence_strength": "strong",
            "consistency": "mostly_consistent",
        },
        "policy": {
            "mode": "strict_grounded",
            "notes": ["exact_event_retrieval"],
        },
    }

    record = build_live_run_record(
        query=response["query"],
        status="success",
        error=None,
        response=response,
        trace=_fake_trace(),
        latency_ms=12.5,
        request_meta={"method": "POST", "path": "/recommend"},
    )

    assert record.run_id
    assert record.status == "success"
    assert record.failure_kind is None
    assert record.resolved_event_name == "Atlas Mountain Race"
    assert record.retrieval_source == "exact_event"
    assert record.retrieval_mode == "exact_only"
    assert record.policy_mode == "strict_grounded"
    assert record.setup_complete is True
    assert record.missing_fields == []
    assert record.component_relevance["passed"] is True
    assert record.quality_issue_codes == []
    assert record.request_meta["method"] == "POST"


def test_build_live_run_record_flags_fallback_honesty_issues():
    response = {
        "query": "What drivetrain should I use for a bikepacking race?",
        "resolved_event": {
            "display_name": "Atlas Mountain Race",
            "canonical_name": "Atlas Mountain Race",
            "match_type": "exact",
            "is_trusted_exact": True,
        },
        "intent": {"component": "drivetrain", "confidence": 0.88},
        "recommendation": {
            "event": "Atlas Mountain Race",
            "summary": "Atlas Mountain Race riders often use a 1x drivetrain.",
            "reasoning": "These exact event riders show consistent gearing choices.",
            "recommended_setup": {
                "drivetrain": "GRX 1x",
                "bike_type": "gravel bike",
            },
        },
        "evidence": {
            "rider_count": 1,
            "component_hit_count": 1,
            "evidence_strength": "moderate",
            "consistency": "sparse",
        },
        "policy": {
            "mode": "strict_grounded",
            "notes": ["unknown_event"],
        },
    }
    trace = SimpleNamespace(
        calls=[
            {
                "tool": "policy_selection",
                "args": {
                    "event_match_type": "exact",
                    "matched_event_name": "Atlas Mountain Race",
                    "retrieval_source": "unknown_global",
                    "exact_event_hit_count": 0,
                },
                "result": {
                    "mode": "strict_grounded",
                    "notes": ["unknown_event"],
                },
                "elapsed_ms": 0.0,
            }
        ]
    )

    record = build_live_run_record(
        query=response["query"],
        status="success",
        error=None,
        response=response,
        trace=trace,
        latency_ms=9.2,
        request_meta={},
    )

    assert record.retrieval_source == "unknown_global"
    assert record.retrieval_mode == "generic_fallback"
    assert "unknown_global_with_strict_grounded_policy" in record.retrieval_policy_issues
    assert "similar_event_fallback_not_disclosed" in record.retrieval_policy_issues


def test_build_live_run_record_marks_guard_blocks():
    trace = SimpleNamespace(
        calls=[
            {
                "tool": "event_resolution",
                "args": {"user_query": "What fee for a burger king"},
                "result": {
                    "display_name": "Unknown event",
                    "match_type": "unknown",
                    "confidence": 0.0,
                    "is_trusted_exact": False,
                },
                "elapsed_ms": 0.0,
            },
            {
                "tool": "intent_classification",
                "args": {"user_query": "What fee for a burger king"},
                "result": {"component": "full_setup", "confidence": 0.25, "component_terms": []},
                "elapsed_ms": 0.0,
            },
            {
                "tool": "guard_decision",
                "args": {"user_query": "What fee for a burger king"},
                "result": {
                    "guard_type": "out_of_domain",
                    "allow_recommendation": False,
                    "reason": "Query does not appear to be about bikepacking or a known event.",
                    "allow_exact_grounding": False,
                    "user_message": "This query does not appear to be a bikepacking event or setup request.",
                },
                "elapsed_ms": 0.0,
            },
        ]
    )
    response = {
        "query": "What fee for a burger king",
        "status": "skipped",
        "message": "This query does not appear to be a bikepacking event or setup request.",
        "guard": {
            "guard_type": "out_of_domain",
            "allow_recommendation": False,
            "reason": "Query does not appear to be about bikepacking or a known event.",
            "allow_exact_grounding": False,
            "user_message": "This query does not appear to be a bikepacking event or setup request.",
        },
    }

    record = build_live_run_record(
        query=response["query"],
        status="skipped",
        error=None,
        response=response,
        trace=trace,
        latency_ms=2.0,
        request_meta={"method": "POST"},
    )

    assert record.status == "skipped"
    assert record.failure_kind == "guard_blocked"
    assert record.guard_type == "out_of_domain"
    assert record.retrieval_source == ""
    assert record.policy_mode == ""
    assert record.setup_complete is False


def test_classify_failure_kind_matches_schema_errors():
    assert classify_failure_kind("ValidationError: output schema failure") == "output_schema_failure"
    assert classify_failure_kind("boom") == "runtime_failure"


def test_api_service_persists_live_eval_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import baikpacking.api.service as service_mod
    from baikpacking.agents import live_evaluation as live_mod

    monkeypatch.setattr(live_mod, "DEFAULT_LIVE_RUNS_PATH", tmp_path / "live_runs.jsonl")

    fake_recommendation = SetupRecommendation(
        event="Atlas Mountain Race",
        summary="Use a grounded tyre setup.",
        reasoning="Exact event evidence supports this.",
        recommended_setup=SetupCore(
            bike_type="gravel bike",
            wheels="700c",
            tyres="45mm semi-slick",
            drivetrain="GRX 1x",
            bags="frame bag",
            sleep_system="bivy",
        ),
    )
    fake_trace = SimpleNamespace(
        calls=[
            {
                "tool": "event_resolution",
                "args": {"user_query": "What tyres do you recommend for Atlas Mountain Race?"},
                "result": {
                    "display_name": "Atlas Mountain Race",
                    "canonical_name": "Atlas Mountain Race",
                    "match_type": "exact",
                    "is_trusted_exact": True,
                },
                "elapsed_ms": 0.0,
            },
            {
                "tool": "intent_classification",
                "args": {"user_query": "What tyres do you recommend for Atlas Mountain Race?"},
                "result": {"component": "tyres", "confidence": 0.92, "component_terms": ["tyre", "tyres"]},
                "elapsed_ms": 0.0,
            },
            {
                "tool": "evidence_summary",
                "args": {},
                "result": {
                    "rider_count": 3,
                    "component_hit_count": 2,
                    "field_support": {"tyres": "pattern"},
                    "evidence_strength": "strong",
                    "consistency": "mostly_consistent",
                },
                "elapsed_ms": 0.0,
            },
            {
                "tool": "policy_selection",
                "args": {
                    "event_match_type": "exact",
                    "matched_event_name": "Atlas Mountain Race",
                    "retrieval_source": "exact_event",
                    "exact_event_hit_count": 2,
                },
                "result": {
                    "mode": "strict_grounded",
                    "allow_specific_brands": True,
                    "allow_specific_specs": True,
                    "allow_event_specific_claims": True,
                    "notes": ["exact_event_retrieval"],
                },
                "elapsed_ms": 0.0,
            },
        ]
    )

    def fake_run(user_query: str):
        return fake_recommendation, fake_trace

    monkeypatch.setattr(service_mod, "recommend_setup_with_trace", fake_run)

    service = service_mod.RecommendationService()
    request = service_mod.RecommendRequest(query="What tyres do you recommend for Atlas Mountain Race?", include_debug=False)
    response = asyncio.run(
        service.recommend(
            request,
            request_meta={"method": "POST", "path": "/recommend", "request_id": "abc123"},
        )
    )

    assert response.recommendation.summary == "Use a grounded tyre setup."

    live_path = tmp_path / "live_runs.jsonl"
    assert live_path.exists()
    rows = [json.loads(line) for line in live_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "What tyres do you recommend for Atlas Mountain Race?"
    assert row["retrieval_source"] == "exact_event"
    assert row["retrieval_mode"] == "exact_only"
    assert row["policy_mode"] == "strict_grounded"
    assert row["setup_complete"] is True
    assert row["quality_issue_codes"] == []
