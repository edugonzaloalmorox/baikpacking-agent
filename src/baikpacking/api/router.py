"""API routes for the bikepacking recommender."""

from __future__ import annotations

import json
import threading
from queue import Queue

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from baikpacking.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendationStreamEvent,
    ReadyResponse,
)
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


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest,
    http_request: Request,
    service: RecommendationService = Depends(get_recommendation_service),
) -> FeedbackResponse:
    """Store a user feedback event for a prior recommendation run."""
    request_meta = {
        "method": http_request.method,
        "path": http_request.url.path,
        "client_host": http_request.client.host if http_request.client else None,
        "user_agent": http_request.headers.get("user-agent"),
        "request_id": http_request.headers.get("x-request-id") or http_request.headers.get("x-correlation-id"),
    }
    return await service.submit_feedback(request, request_meta=request_meta)


@router.post("/recommend/stream")
async def recommend_stream(
    request: RecommendRequest,
    http_request: Request,
    service: RecommendationService = Depends(get_recommendation_service),
) -> StreamingResponse:
    """Stream recommendation progress and the final result as NDJSON."""
    request_meta = {
        "method": http_request.method,
        "path": http_request.url.path,
        "client_host": http_request.client.host if http_request.client else None,
        "user_agent": http_request.headers.get("user-agent"),
        "request_id": http_request.headers.get("x-request-id") or http_request.headers.get("x-correlation-id"),
        "include_debug": request.include_debug,
    }

    queue: Queue[object] = Queue()
    sentinel = object()

    def _emit_progress(progress) -> None:
        queue.put(
            RecommendationStreamEvent(
                kind="progress",
                progress=progress,
            ).model_dump(mode="json")
        )

    def _worker() -> None:
        try:
            response = service.recommend_sync(request, request_meta=request_meta, progress_callback=_emit_progress)
            queue.put(
                RecommendationStreamEvent(
                    kind="final",
                    response=response.model_dump(mode="json"),
                ).model_dump(mode="json")
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500)
            queue.put(
                RecommendationStreamEvent(
                    kind="error",
                    error=str(exc),
                    status_code=status_code,
                ).model_dump(mode="json")
            )
        finally:
            queue.put(sentinel)

    threading.Thread(target=_worker, daemon=True).start()

    async def _iter_lines():
        while True:
            item = await anyio.to_thread.run_sync(queue.get)
            if item is sentinel:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(_iter_lines(), media_type="application/x-ndjson")
