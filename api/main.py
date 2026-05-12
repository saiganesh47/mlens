"""
api/main.py
============
MLens REST API — FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health          → health check
    GET  /version         → version info
    POST /audit           → run full ML audit
    POST /audit/shap      → SHAP only
    POST /audit/fairness  → fairness only
    POST /audit/drift     → drift only
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import audit, health

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "MLens API",
    description = "Explainable ML Audit Tool — SHAP, Fairness & Drift Detection",
    version     = "0.3.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(health.router, tags=["Health"])
app.include_router(audit.router,  tags=["Audit"], prefix="/audit")

# ── Root ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name":        "MLens API",
        "version":     "0.3.0",
        "description": "Explainable ML Audit Tool",
        "docs":        "/docs",
        "github":      "https://github.com/saiganesh47/mlens",
        "endpoints": {
            "health":          "GET  /health",
            "full_audit":      "POST /audit",
            "shap_only":       "POST /audit/shap",
            "fairness_only":   "POST /audit/fairness",
            "drift_only":      "POST /audit/drift",
        },
    }


# ── Global exception handler ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error":   type(exc).__name__,
            "message": str(exc),
            "hint":    "Check /docs for correct request format.",
        },
    )
