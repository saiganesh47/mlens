"""
api/routes/monitor.py
======================
REST API endpoints for real-time model monitoring.

POST /monitor/ingest   → ingest a prediction batch
GET  /monitor/status   → current health snapshot
GET  /monitor/history  → last N snapshots
POST /monitor/reset    → reset the monitor window
GET  /monitor/models   → list all registered monitors
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory monitor registry {model_name: RealtimeMonitor}
_monitors: Dict[str, Any] = {}


# ── Request / Response schemas ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    model_name         : str
    y_true             : List[float]
    y_pred             : List[float]
    sensitive_features : Optional[List[Any]] = None
    window_size        : int   = Field(500, ge=10, le=10000)
    error_threshold    : float = Field(0.20, ge=0.0, le=1.0)
    dp_gap_threshold   : float = Field(0.10, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name":         "MyProductionModel",
                "y_true":             [1, 0, 1, 1, 0],
                "y_pred":             [1, 0, 0, 1, 0],
                "sensitive_features": ["M", "F", "M", "F", "M"],
            }
        }
    }


class SnapshotResponse(BaseModel):
    model_name    : str
    timestamp     : str
    n_predictions : int
    error_rate    : float
    drift_detected: bool
    dp_gap        : Optional[float]
    alert_count   : int
    window_size   : int
    health        : str    # 'healthy' | 'warning' | 'critical'


class HistoryResponse(BaseModel):
    model_name : str
    n_snapshots: int
    snapshots  : List[Dict[str, Any]]


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_or_create_monitor(
    model_name      : str,
    window_size     : int   = 500,
    error_threshold : float = 0.20,
    dp_gap_threshold: float = 0.10,
) -> Any:
    """Get existing monitor or create a new one."""
    from mlens.monitoring.realtime_monitor import RealtimeMonitor
    if model_name not in _monitors:
        _monitors[model_name] = RealtimeMonitor(
            model_name           = model_name,
            window_size          = window_size,
            error_rate_threshold = error_threshold,
            dp_gap_threshold     = dp_gap_threshold,
        )
    return _monitors[model_name]


def _health_label(snap: Any) -> str:
    if snap.drift_detected or snap.error_rate > 0.30:
        return "critical"
    if snap.error_rate > 0.20 or (snap.dp_gap and snap.dp_gap > 0.10):
        return "warning"
    return "healthy"


def _snap_to_response(model_name: str, snap: Any) -> SnapshotResponse:
    return SnapshotResponse(
        model_name     = model_name,
        timestamp      = snap.timestamp,
        n_predictions  = snap.n_predictions,
        error_rate     = snap.error_rate,
        drift_detected = snap.drift_detected,
        dp_gap         = snap.dp_gap,
        alert_count    = snap.alert_count,
        window_size    = snap.window_size,
        health         = _health_label(snap),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=SnapshotResponse,
    summary="Ingest Prediction Batch",
    description="""
Ingest a batch of predictions into the rolling monitor window.

Returns the current health snapshot after ingestion including
error rate, drift status, and demographic parity gap.
    """,
)
async def ingest_predictions(req: IngestRequest) -> SnapshotResponse:
    try:
        import numpy as np
        monitor = _get_or_create_monitor(
            req.model_name, req.window_size,
            req.error_threshold, req.dp_gap_threshold,
        )
        snap = monitor.ingest(
            y_true    = np.array(req.y_true),
            y_pred    = np.array(req.y_pred),
            sensitive = req.sensitive_features,
        )
        return _snap_to_response(req.model_name, snap)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/status/{model_name}",
    response_model=SnapshotResponse,
    summary="Get Monitor Status",
)
async def get_status(model_name: str) -> SnapshotResponse:
    if model_name not in _monitors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No monitor registered for '{model_name}'. "
                   f"POST to /monitor/ingest first.",
        )
    monitor = _monitors[model_name]
    snap    = monitor.status()
    return _snap_to_response(model_name, snap)


@router.get(
    "/history/{model_name}",
    response_model=HistoryResponse,
    summary="Get Monitor History",
)
async def get_history(model_name: str, n: int = 50) -> HistoryResponse:
    if model_name not in _monitors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No monitor found for '{model_name}'.",
        )
    history = _monitors[model_name].history(n=n)
    return HistoryResponse(
        model_name  = model_name,
        n_snapshots = len(history),
        snapshots   = [s.to_dict() for s in history],
    )


@router.post(
    "/reset/{model_name}",
    summary="Reset Monitor Window",
)
async def reset_monitor(model_name: str) -> Dict[str, str]:
    if model_name not in _monitors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No monitor found for '{model_name}'.",
        )
    _monitors[model_name].reset()
    return {"status": "reset", "model": model_name}


@router.get(
    "/models",
    summary="List Monitored Models",
)
async def list_models() -> Dict[str, Any]:
    result = {}
    for name, monitor in _monitors.items():
        snap = monitor.status()
        result[name] = {
            "n_predictions": snap.n_predictions,
            "error_rate":    round(snap.error_rate, 4),
            "health":        _health_label(snap),
            "alert_count":   snap.alert_count,
        }
    return {"n_models": len(result), "models": result}
