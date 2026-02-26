"""Error handling and CORS middleware."""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from rezaa.models.errors import ErrorResponse, RezaaError

logger = logging.getLogger(__name__)


async def rezaa_error_handler(request: Request, exc: RezaaError) -> JSONResponse:
    """Handle RezaaError exceptions."""
    response = ErrorResponse(
        error_type=type(exc).__name__,
        component=exc.component,
        message=exc.message,
        details=exc.details,
        actionable_guidance=_get_guidance(exc),
        retry_possible=_is_retryable(exc),
    )
    status_code = _get_status_code(exc)
    return JSONResponse(status_code=status_code, content=response.model_dump())


def _get_status_code(exc: RezaaError) -> int:
    """Map error type to HTTP status code."""
    from rezaa.models.errors import (
        ProcessingError,
        RenderingError,
        ResourceError,
        ValidationError,
    )

    if isinstance(exc, ValidationError):
        return 400
    elif isinstance(exc, ResourceError):
        return 503
    elif isinstance(exc, (ProcessingError, RenderingError)):
        return 500
    return 500


def _get_guidance(exc: RezaaError) -> str:
    """Generate actionable guidance based on error type."""
    from rezaa.models.errors import ValidationError

    if isinstance(exc, ValidationError):
        return "Check your input file format, size, and integrity."
    return "Please try again or contact support."


def _is_retryable(exc: RezaaError) -> bool:
    """Determine if the error is retryable."""
    from rezaa.models.errors import ResourceError

    return isinstance(exc, ResourceError)
