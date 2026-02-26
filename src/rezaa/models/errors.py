"""Error hierarchy and error response models."""

from pydantic import BaseModel, Field


class RezaaError(Exception):
    """Base error for all Rezaa AI errors."""

    def __init__(self, message: str, component: str = "", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.component = component
        self.details = details or {}


class ValidationError(RezaaError):
    """Input validation errors (format, size, integrity)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, component="validation", details=details)


class ProcessingError(RezaaError):
    """Errors during feature extraction or agent analysis."""

    def __init__(self, message: str, component: str = "processing", details: dict | None = None):
        super().__init__(message, component=component, details=details)


class ResourceError(RezaaError):
    """Resource-related errors (disk, memory, external services)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, component="resource", details=details)


class RenderingError(RezaaError):
    """Errors during video rendering."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, component="rendering", details=details)


class ErrorResponse(BaseModel):
    """Standardized error response for API."""

    error_type: str = Field(..., description="Error category")
    component: str = Field(default="", description="Component that raised the error")
    message: str = Field(..., description="Human-readable error message")
    details: dict = Field(default_factory=dict)
    actionable_guidance: str = Field(default="", description="Suggested user action")
    retry_possible: bool = Field(default=False)

    @classmethod
    def from_exception(
        cls, exc: RezaaError, guidance: str = "", retry: bool = False
    ) -> "ErrorResponse":
        return cls(
            error_type=type(exc).__name__,
            component=exc.component,
            message=exc.message,
            details=exc.details,
            actionable_guidance=guidance,
            retry_possible=retry,
        )
