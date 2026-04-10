import json

from fastapi.testclient import TestClient

from baikpacking.api.main import create_app
from baikpacking.api.router import get_recommendation_service
from baikpacking.api.schemas import RecommendResponse
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
    assert payload["query"] == "What tyres should I use for Atlas Mountain Race?"
    assert payload["resolved_event"]["display_name"] == "Atlas Mountain Race"
    assert payload["recommendation"]["summary"] == "Use a grounded tyre setup."
    assert payload["policy"]["mode"] == "strict_grounded"


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
