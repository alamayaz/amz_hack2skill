"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rezaa.api.middleware import rezaa_error_handler
from rezaa.api.routes import download, process, status, upload
from rezaa.models.errors import RezaaError


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Rezaa AI",
        description="Intelligent multi-agent video editing system",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    app.add_exception_handler(RezaaError, rezaa_error_handler)

    # Routes
    app.include_router(upload.router)
    app.include_router(process.router)
    app.include_router(status.router)
    app.include_router(download.router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    # Static files — mount AFTER API routes so /api/v1/* takes priority
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
