"""FastAPI application factory for the bikepacking recommender."""


import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from baikpacking.api.router import router
from baikpacking.api.schemas import ErrorResponse
from baikpacking.api.service import ApiServiceError
from baikpacking.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the HTTP API app."""
    setup_logging()

    app = FastAPI(
        title="bAIpacking API",
        description="HTTP API around the bikepacking recommender pipeline.",
        version="0.1.0",
    )
    app.include_router(router)

    @app.exception_handler(ApiServiceError)
    async def _handle_api_service_error(_: Request, exc: ApiServiceError) -> JSONResponse:
        logger.warning("api_service_error status=%s message=%s", exc.status_code, str(exc))
        payload = ErrorResponse(error="api_service_error", detail=str(exc)).model_dump()
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(error="validation_error", detail="Invalid request body").model_dump()
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected_api_error")
        payload = ErrorResponse(error="internal_error", detail="Unexpected server error").model_dump()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload)

    return app


app = create_app()
