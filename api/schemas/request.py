"""
api/schemas/request.py
=======================
Pydantic v2 models for all API request and response bodies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request models ─────────────────────────────────────────────────────────

class AuditRequest(BaseModel):
    """
    Full audit request body.

    Fields
    ------
    X_train : list of lists
        Training feature matrix (rows × features).
        Used as reference for drift detection and SHAP background.
    X_test : list of lists
        Test feature matrix.
    y_test : list of int/float
        Ground-truth labels for X_test.
    sensitive_features : list of str/int, optional
        Protected attribute values aligned with X_test rows.
    feature_names : list of str, optional
        Human-readable column names.
    model_name : str
        Display name for the model in the report.
    run_shap : bool
        Toggle SHAP explainability (default True).
    run_fairness : bool
        Toggle fairness evaluation (default True).
    run_drift : bool
        Toggle drift detection (default True).
    shap_background_samples : int
        Rows sampled from X_train for SHAP background (default 100).
    """

    X_train               : List[List[float]] = Field(..., description="Training feature matrix")
    X_test                : List[List[float]] = Field(..., description="Test feature matrix")
    y_test                : List[float]        = Field(..., description="Test labels")
    sensitive_features    : Optional[List[Any]] = Field(None, description="Protected attribute column")
    feature_names         : Optional[List[str]] = Field(None, description="Feature column names")
    model_name            : str                 = Field("MLensModel", description="Model display name")
    run_shap              : bool                = True
    run_fairness          : bool                = True
    run_drift             : bool                = True
    shap_background_samples: int               = Field(100, ge=10, le=500)

    @field_validator("X_test")
    @classmethod
    def x_test_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("X_test must not be empty.")
        return v

    @field_validator("y_test")
    @classmethod
    def y_test_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("y_test must not be empty.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "X_train": [[25, 50000, 40], [35, 80000, 45], [45, 30000, 35]],
                "X_test":  [[30, 60000, 38], [50, 20000, 30]],
                "y_test":  [1, 0],
                "sensitive_features": ["Male", "Female"],
                "feature_names": ["age", "income", "hours_per_week"],
                "model_name": "MyProductionModel",
            }
        }
    }


class SHAPRequest(BaseModel):
    """Request body for SHAP-only endpoint."""
    X_train       : List[List[float]]
    X_test        : List[List[float]]
    feature_names : Optional[List[str]] = None
    model_name    : str = "MLensModel"
    background_samples: int = Field(100, ge=10, le=500)


class FairnessRequest(BaseModel):
    """Request body for fairness-only endpoint."""
    y_true             : List[float]
    y_pred             : List[float]
    sensitive_features : List[Any]
    sensitive_feature_name: str = "sensitive_feature"
    dp_threshold       : float = Field(0.10, ge=0.0, le=1.0)
    eo_threshold       : float = Field(0.10, ge=0.0, le=1.0)
    di_threshold       : float = Field(0.80, ge=0.0, le=1.0)


class DriftRequest(BaseModel):
    """Request body for drift-only endpoint."""
    X_reference   : List[List[float]]
    X_production  : List[List[float]]
    feature_names : Optional[List[str]] = None
    psi_bins      : int   = Field(10, ge=5, le=50)
    ks_alpha      : float = Field(0.05, ge=0.001, le=0.5)


# ── Response models ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status    : str
    version   : str
    timestamp : str


class TopFeature(BaseModel):
    rank          : int
    name          : str
    mean_abs_shap : float


class SHAPResponse(BaseModel):
    model_name   : str
    base_value   : float
    top_features : List[TopFeature]


class FairnessResponse(BaseModel):
    sensitive_feature        : str
    demographic_parity_gap   : float
    equalized_odds_gap       : float
    disparate_impact         : float
    is_fair                  : bool
    flags                    : List[str]
    per_group_metrics        : List[Dict[str, Any]]


class DriftFeature(BaseModel):
    feature       : str
    psi           : float
    psi_status    : str
    ks_statistic  : float
    ks_pvalue     : float
    drifted       : bool


class DriftResponse(BaseModel):
    overall_status   : str
    n_drifted        : int
    drifted_features : List[str]
    details          : List[DriftFeature]


class AuditResponse(BaseModel):
    model_name       : str
    audit_timestamp  : str
    runtime_seconds  : float
    summary          : List[str]
    metadata         : Dict[str, Any]
    shap             : Optional[SHAPResponse]    = None
    fairness         : Optional[FairnessResponse] = None
    drift            : Optional[DriftResponse]   = None
