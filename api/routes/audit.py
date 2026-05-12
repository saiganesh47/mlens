"""
api/routes/audit.py
====================
Core audit endpoints.

POST /audit           → full audit (SHAP + Fairness + Drift)
POST /audit/shap      → SHAP explainability only
POST /audit/fairness  → fairness evaluation only
POST /audit/drift     → drift detection only
"""

from __future__ import annotations

import pickle
import tempfile
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, status

from api.schemas.request import (
    AuditRequest, AuditResponse,
    DriftRequest, DriftResponse,
    FairnessRequest, FairnessResponse,
    SHAPRequest, SHAPResponse,
    TopFeature, DriftFeature,
)

router = APIRouter()


# ── Helper: build a fake model from data for demo purposes ─────────────────

def _get_model_from_request(X_train, y_train=None):
    """
    In production, the model would be loaded from a model registry
    (MLflow, S3, etc). For demo, we train a fast GBT on the fly.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, random_state=42
    )
    if y_train is None:
        # Create dummy labels for background-only use
        y_train = np.random.randint(0, 2, len(X_train))
    model.fit(X_train, y_train)
    return model


# ── POST /audit ─────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=AuditResponse,
    summary="Run Full ML Audit",
    description="""
Run a complete MLens audit pipeline on your data.

Returns:
- **SHAP** global feature importance + top features
- **Fairness** metrics: demographic parity, equalized odds, disparate impact
- **Drift** detection: PSI + KS-test per feature
- **Summary** plain-English findings
    """,
)
async def run_full_audit(req: AuditRequest) -> AuditResponse:
    try:
        X_train = np.array(req.X_train, dtype=float)
        X_test  = np.array(req.X_test,  dtype=float)
        y_test  = np.array(req.y_test,  dtype=float)

        # Validate shapes
        if X_train.shape[1] != X_test.shape[1]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"X_train has {X_train.shape[1]} features but "
                       f"X_test has {X_test.shape[1]}.",
            )
        if len(X_test) != len(y_test):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="X_test and y_test must have the same number of rows.",
            )

        # Train a model on the provided data
        model = _get_model_from_request(X_train, y_test[:len(X_train)]
                                        if len(y_test) >= len(X_train)
                                        else None)

        from mlens.auditor import ModelAuditor
        auditor = ModelAuditor(
            model              = model,
            X_train            = X_train,
            X_test             = X_test,
            y_test             = y_test,
            sensitive_features = req.sensitive_features,
            feature_names      = req.feature_names,
            model_name         = req.model_name,
            shap_background_samples = req.shap_background_samples,
            run_shap           = req.run_shap,
            run_fairness       = req.run_fairness and req.sensitive_features is not None,
            run_drift          = req.run_drift,
        )
        report = auditor.run()
        return _report_to_response(report)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ── POST /audit/shap ────────────────────────────────────────────────────────

@router.post(
    "/shap",
    response_model=SHAPResponse,
    summary="SHAP Explainability Only",
)
async def run_shap_only(req: SHAPRequest) -> SHAPResponse:
    try:
        X_train = np.array(req.X_train, dtype=float)
        X_test  = np.array(req.X_test,  dtype=float)
        model   = _get_model_from_request(X_train)

        from mlens.explainability.shap_analyzer import ShapAnalyzer
        analyzer = ShapAnalyzer(
            model          = model,
            background_data= X_train[:req.background_samples],
            feature_names  = req.feature_names,
        )
        result = analyzer.explain(X_test)
        return SHAPResponse(
            model_name   = req.model_name,
            base_value   = result.base_value,
            top_features = [
                TopFeature(**f) for f in result.top_features(n=15)
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /audit/fairness ────────────────────────────────────────────────────

@router.post(
    "/fairness",
    response_model=FairnessResponse,
    summary="Fairness Evaluation Only",
)
async def run_fairness_only(req: FairnessRequest) -> FairnessResponse:
    try:
        from mlens.fairness.fairness_metrics import FairnessEvaluator
        evaluator = FairnessEvaluator(
            y_true                = np.array(req.y_true),
            y_pred                = np.array(req.y_pred),
            sensitive_features    = np.array(req.sensitive_features),
            sensitive_feature_name= req.sensitive_feature_name,
            dp_threshold          = req.dp_threshold,
            eo_threshold          = req.eo_threshold,
            di_threshold          = req.di_threshold,
        )
        result = evaluator.evaluate()
        return FairnessResponse(
            sensitive_feature      = result.sensitive_feature_name,
            demographic_parity_gap = result.demographic_parity_gap,
            equalized_odds_gap     = result.equalized_odds_gap,
            disparate_impact       = result.disparate_impact,
            is_fair                = result.is_fair,
            flags                  = result.flags,
            per_group_metrics      = result.per_group_metrics,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /audit/drift ───────────────────────────────────────────────────────

@router.post(
    "/drift",
    response_model=DriftResponse,
    summary="Drift Detection Only",
)
async def run_drift_only(req: DriftRequest) -> DriftResponse:
    try:
        from mlens.drift.drift_detector import DriftDetector
        detector = DriftDetector(
            reference     = np.array(req.X_reference,  dtype=float),
            feature_names = req.feature_names,
            psi_bins      = req.psi_bins,
            ks_alpha      = req.ks_alpha,
        )
        result = detector.detect(np.array(req.X_production, dtype=float))
        return DriftResponse(
            overall_status   = result.overall_status,
            n_drifted        = result.n_drifted,
            drifted_features = result.drifted_features(),
            details          = [DriftFeature(**f) for f in result.feature_results],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Helper ──────────────────────────────────────────────────────────────────

def _report_to_response(report) -> AuditResponse:
    """Convert AuditReport dataclass → AuditResponse Pydantic model."""
    shap_resp = None
    if report.shap_result:
        shap_resp = SHAPResponse(
            model_name   = report.model_name,
            base_value   = report.shap_result.base_value,
            top_features = [
                TopFeature(**f) for f in report.shap_result.top_features(n=15)
            ],
        )

    fair_resp = None
    if report.fairness_result:
        fr = report.fairness_result
        fair_resp = FairnessResponse(
            sensitive_feature      = fr.sensitive_feature_name,
            demographic_parity_gap = fr.demographic_parity_gap,
            equalized_odds_gap     = fr.equalized_odds_gap,
            disparate_impact       = fr.disparate_impact,
            is_fair                = fr.is_fair,
            flags                  = fr.flags,
            per_group_metrics      = fr.per_group_metrics,
        )

    drift_resp = None
    if report.drift_result:
        dr = report.drift_result
        drift_resp = DriftResponse(
            overall_status   = dr.overall_status,
            n_drifted        = dr.n_drifted,
            drifted_features = dr.drifted_features(),
            details          = [DriftFeature(**f) for f in dr.feature_results],
        )

    return AuditResponse(
        model_name      = report.model_name,
        audit_timestamp = report.audit_timestamp,
        runtime_seconds = round(report.runtime_seconds, 3),
        summary         = report.summary_lines,
        metadata        = report.metadata,
        shap            = shap_resp,
        fairness        = fair_resp,
        drift           = drift_resp,
    )
