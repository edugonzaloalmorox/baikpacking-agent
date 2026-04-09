from fastapi.testclient import TestClient

from baikpacking.api.main import create_app
from baikpacking.api.router import get_recommendation_service
from baikpacking.api.schemas import RecommendResponse
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

    async def recommend(self, request):
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
