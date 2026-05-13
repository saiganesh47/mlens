"""
api/routes/health.py
=====================
Health check and version endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from api.schemas.request import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status, version, and current UTC timestamp.
    """
    return HealthResponse(
        status    = "healthy",
        version   = "0.3.0",
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@router.get("/version")
async def version():
    """Return MLens version and build info."""
    return {
        "version":     "0.3.0",
        "release":     "Deploy Anywhere",
        "author":      "github.com/saiganesh47/mlens",
        "python_min":  "3.9",
        "framework":   "FastAPI",
    }
