"""
app.py — FastAPI application factory.

Creates the FastAPI app with CORS middleware and a lifespan handler
that loads the pipeline on startup (so models are warm before the
first request).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline import Pipeline

logger = logging.getLogger(__name__)

# Global pipeline instance — initialized during app lifespan startup
pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Return the global pipeline instance.  Raises if not yet initialized."""
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized — server is still starting up")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan handler: load models on startup, cleanup on shutdown."""
    global pipeline
    logger.info("Starting pipeline initialization...")
    pipeline = Pipeline()
    logger.info("Pipeline ready — server accepting requests")
    yield
    logger.info("Shutting down pipeline")
    pipeline = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Traffic Violation Detection API",
        description=(
            "Dual-model, parallel-inference system for comprehensive "
            "traffic violation detection with license plate OCR."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for hackathon/development flexibility.
    # Tighten this for production deployments.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include route handlers
    from server.routes import router

    app.include_router(router)

    return app
