"""API routes for the bikepacking recommender."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

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
    http_request: Request,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendResponse:
    """Generate a grounded recommendation for the given query."""
    request_meta = {
        "method": http_request.method,
        "path": http_request.url.path,
        "client_host": http_request.client.host if http_request.client else None,
        "user_agent": http_request.headers.get("user-agent"),
        "request_id": http_request.headers.get("x-request-id") or http_request.headers.get("x-correlation-id"),
        "include_debug": request.include_debug,
    }
    return await service.recommend(request, request_meta=request_meta)
