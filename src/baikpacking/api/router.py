"""API routes for the bikepacking recommender."""



from fastapi import APIRouter, Depends

from baikpacking.api.schemas import HealthResponse, RecommendRequest, RecommendResponse, ReadyResponse
from baikpacking.api.service import RecommendationService, get_recommendation_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(service: RecommendationService = Depends(get_recommendation_service)) -> HealthResponse:
    """Return a lightweight liveness response."""
    return await service.health()


@router.get("/ready", response_model=ReadyResponse)
async def ready(service: RecommendationService = Depends(get_recommendation_service)) -> ReadyResponse:
    """Return readiness after a database connectivity check."""
    return await service.ready()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: RecommendRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendResponse:
    """Generate a grounded recommendation for the given query."""
    return await service.recommend(request)
