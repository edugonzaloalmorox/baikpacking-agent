import json

from fastapi.testclient import TestClient

from baikpacking.api.main import create_app
from baikpacking.api.router import get_recommendation_service
from baikpacking.api.schemas import FeedbackResponse, RecommendResponse
from baikpacking.agents.guardrails import GuardDecision, RecommendationGuardBlocked
from baikpacking.agents.models import QueryIntent
from baikpacking.agents.models import SetupCore, SetupRecommendation
from baikpacking.agents.orchestration_models import EvidenceSummary, EventResolutionResult, RecommendationPolicy


class _FakeService:
    async def health(self):
        from baikpacking.api.schemas import HealthResponse

        return HealthResponse()

    async def ready(self):
        from baikpacking.api.schemas import ReadyResponse

        return ReadyResponse(database={"database": "baikpacking"})

    async def recommend(self, request, request_meta=None):
        return RecommendResponse(
            run_id="run-123",
            query=request.query,
            resolved_event=EventResolutionResult(
                display_name="Atlas Mountain Race",
                canonical_name="Atlas Mountain Race",
                match_type="exact",
                confidence=0.98,
                is_trusted_exact=True,
            ),
            intent=QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"]),
            recommendation=SetupRecommendation(
                event="Atlas Mountain Race",
                summary="Use a grounded tyre setup.",
                reasoning="Exact event evidence supports this.",
                recommended_setup=SetupCore(tyres="45mm semi-slick"),
                similar_riders=[],
            ),
            evidence=EvidenceSummary(
                rider_count=3,
                component_hit_count=2,
                field_support={"tyres": "pattern"},
                evidence_strength="strong",
                consistency="mostly_consistent",
            ),
            policy=RecommendationPolicy(
                mode="strict_grounded",
                allow_specific_brands=True,
                allow_specific_specs=True,
                allow_event_specific_claims=True,
                notes=["exact_event_retrieval"],
            ),
            debug=None,
        )

    def recommend_sync(self, request, request_meta=None, progress_callback=None):
        from baikpacking.agents.progress import build_progress_event

        if progress_callback is not None:
            for stage_key, stage_label in [
                ("resolving_event", "Resolving event"),
                ("classifying_intent", "Classifying intent"),
                ("searching_riders", "Searching riders"),
                ("selecting_policy", "Selecting policy"),
                ("writing_recommendation", "Writing recommendation"),
            ]:
                progress_callback(build_progress_event(stage_key, stage_label))
        return self._build_recommendation(request)

    def _build_recommendation(self, request):
        return RecommendResponse(
            run_id="run-123",
            query=request.query,
            resolved_event=EventResolutionResult(
                display_name="Atlas Mountain Race",
                canonical_name="Atlas Mountain Race",
                match_type="exact",
                confidence=0.98,
                is_trusted_exact=True,
            ),
            intent=QueryIntent(component="tyres", confidence=0.9, component_terms=["tyre", "tyres"]),
            recommendation=SetupRecommendation(
                event="Atlas Mountain Race",
                summary="Use a grounded tyre setup.",
                reasoning="Exact event evidence supports this.",
                recommended_setup=SetupCore(tyres="45mm semi-slick"),
                similar_riders=[],
            ),
            evidence=EvidenceSummary(
                rider_count=3,
                component_hit_count=2,
                field_support={"tyres": "pattern"},
                evidence_strength="strong",
                consistency="mostly_consistent",
            ),
            policy=RecommendationPolicy(
                mode="strict_grounded",
                allow_specific_brands=True,
                allow_specific_specs=True,
                allow_event_specific_claims=True,
                notes=["exact_event_retrieval"],
            ),
            debug=None,
        )

    async def submit_feedback(self, request, request_meta=None):
        return FeedbackResponse(run_id=request.run_id, feedback=request.feedback, timestamp="2026-04-10T00:00:00+00:00")


def _make_client():
    app = create_app()
    app.dependency_overrides[get_recommendation_service] = lambda: _FakeService()
    return TestClient(app)


def test_health_endpoint():
    client = _make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_endpoint():
    client = _make_client()

    response = client.post(
        "/recommend",
        json={"query": "What tyres should I use for Atlas Mountain Race?", "include_debug": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-123"
    assert payload["query"] == "What tyres should I use for Atlas Mountain Race?"
    assert payload["resolved_event"]["display_name"] == "Atlas Mountain Race"
    assert payload["recommendation"]["summary"] == "Use a grounded tyre setup."
    assert payload["policy"]["mode"] == "strict_grounded"


def test_feedback_endpoint():
    client = _make_client()

    response = client.post(
        "/feedback",
        json={"run_id": "run-123", "feedback": "thumbs_down", "comment": "Wrong event"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-123"
    assert payload["feedback"] == "thumbs_down"


def test_recommend_stream_endpoint():
    client = _make_client()

    with client.stream(
        "POST",
        "/recommend/stream",
        json={"query": "What tyres should I use for Atlas Mountain Race?", "include_debug": True},
    ) as response:
        assert response.status_code == 200
        lines = [json.loads(line) for line in response.iter_lines() if line.strip()]

    progress_events = [line for line in lines if line["kind"] == "progress"]
    assert [item["progress"]["stage_key"] for item in progress_events] == [
        "resolving_event",
        "classifying_intent",
        "searching_riders",
        "selecting_policy",
        "writing_recommendation",
    ]
    final_events = [line for line in lines if line["kind"] == "final"]
    assert final_events and final_events[0]["response"]["run_id"] == "run-123"


def test_recommend_endpoint_returns_skipped_response_for_guard_block(tmp_path, monkeypatch):
    from baikpacking.agents import live_evaluation as live_mod
    import baikpacking.api.service as service_mod

    monkeypatch.setattr(live_mod, "DEFAULT_LIVE_RUNS_PATH", tmp_path / "live_runs.jsonl")

    decision = GuardDecision(
        allow_recommendation=False,
        reason="Query does not appear to be about bikepacking or a known event.",
        guard_type="out_of_domain",
        allow_exact_grounding=False,
        user_message="This query does not appear to be a bikepacking event or setup request.",
    )
    trace = type("Trace", (), {"calls": [
        {"tool": "guard_decision", "args": {"user_query": "What fee for a burger king"}, "result": decision.model_dump(), "elapsed_ms": 0.0}
    ]})()

    def fake_run(query: str):
        raise RecommendationGuardBlocked(decision, trace=trace)

    monkeypatch.setattr(service_mod, "recommend_setup_with_trace", fake_run)
    client = TestClient(create_app())

    response = client.post("/recommend", json={"query": "What fee for a burger king", "include_debug": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "skipped"
    assert payload["message"] == "This query does not appear to be a bikepacking event or setup request."
    assert payload["guard"]["guard_type"] == "out_of_domain"

    live_path = tmp_path / "live_runs.jsonl"
    assert live_path.exists()
    row = live_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    row_data = json.loads(row)
    assert row_data["status"] == "skipped"
    assert row_data["failure_kind"] == "guard_blocked"
