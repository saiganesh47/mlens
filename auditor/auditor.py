"""
mlens/auditor.py
================
Core audit orchestrator. Accepts any sklearn-compatible, XGBoost, or
PyTorch model and produces a structured AuditReport containing
explainability, fairness, and drift diagnostics.

Usage
-----
>>> from mlens.auditor import ModelAuditor
>>> auditor = ModelAuditor(model, X_train, X_test, y_test,
...                        sensitive_features=df["gender"],
...                        feature_names=feature_names)
>>> report = auditor.run()
>>> report.save("audit_report.html")
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlens.drift.drift_detector import DriftDetector, DriftResult
from mlens.explainability.shap_analyzer import ShapAnalyzer, ShapResult
from mlens.fairness.fairness_metrics import FairnessEvaluator, FairnessResult
from mlens.report.html_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Audit Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """
    Structured container for all audit results.
    Attributes are populated by ModelAuditor.run().
    """

    model_name: str
    audit_timestamp: str
    runtime_seconds: float

    # Sub-reports (None if that module was skipped)
    shap_result: Optional[ShapResult] = None
    fairness_result: Optional[FairnessResult] = None
    drift_result: Optional[DriftResult] = None

    # Plain-English summary lines generated post-analysis
    summary_lines: List[str] = field(default_factory=list)

    # Raw metadata forwarded from the auditor
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ I/O

    def save(self, path: Union[str, Path] = "mlens_report.html") -> Path:
        """Render the report to an HTML file and return the resolved path."""
        out = Path(path).resolve()
        generator = ReportGenerator(self)
        generator.render(out)
        print(f"[MLens] Report saved → {out}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dict (JSON-safe subset)."""
        return {
            "model_name": self.model_name,
            "audit_timestamp": self.audit_timestamp,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "summary": self.summary_lines,
            "fairness": self.fairness_result.to_dict() if self.fairness_result else None,
            "drift": self.drift_result.to_dict() if self.drift_result else None,
            "shap_top_features": (
                self.shap_result.top_features(n=10) if self.shap_result else None
            ),
        }


# ---------------------------------------------------------------------------
# Main Auditor
# ---------------------------------------------------------------------------

class ModelAuditor:
    """
    One-stop ML audit pipeline.

    Parameters
    ----------
    model : sklearn-compatible estimator | XGBClassifier | torch.nn.Module
        The trained model to audit.
    X_train : array-like of shape (n_samples, n_features)
        Training data used as the *reference* distribution for drift detection
        and SHAP background.
    X_test : array-like of shape (n_samples, n_features)
        Held-out evaluation data.
    y_test : array-like of shape (n_samples,)
        Ground-truth labels for X_test.
    sensitive_features : pd.Series | array-like, optional
        Demographic column(s) used for fairness evaluation (e.g. gender, race).
    feature_names : list of str, optional
        Human-readable feature names. Auto-inferred from a DataFrame if passed.
    model_name : str
        Display name embedded in the report (default: class name of model).
    shap_background_samples : int
        Number of rows sampled from X_train for the SHAP explainer background.
    run_drift : bool
        Toggle drift detection (default: True).
    run_fairness : bool
        Toggle fairness evaluation (default: True, requires sensitive_features).
    run_shap : bool
        Toggle SHAP explainability (default: True).
    """

    def __init__(
        self,
        model: Any,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        sensitive_features: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        shap_background_samples: int = 100,
        run_drift: bool = True,
        run_fairness: bool = True,
        run_shap: bool = True,
    ) -> None:
        self.model = model
        self.X_train = self._to_numpy(X_train)
        self.X_test = self._to_numpy(X_test)
        self.y_test = np.asarray(y_test)
        self.sensitive_features = sensitive_features
        self.feature_names = feature_names or self._infer_feature_names(X_train)
        self.model_name = model_name or type(model).__name__
        self.shap_background_samples = shap_background_samples
        self.run_drift = run_drift
        self.run_fairness = run_fairness and sensitive_features is not None
        self.run_shap = run_shap

    # ---------------------------------------------------------------- public

    def run(self) -> AuditReport:
        """
        Execute the full audit pipeline and return an AuditReport.

        The pipeline runs in three phases:
          1. SHAP explainability  — global + local feature importance
          2. Fairness evaluation  — demographic-parity & equalized-odds gaps
          3. Drift detection      — PSI + KS test per feature

        Returns
        -------
        AuditReport
        """
        t0 = time.perf_counter()
        timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n[MLens] Starting audit of '{self.model_name}' …")

        shap_result = self._run_shap()
        fairness_result = self._run_fairness()
        drift_result = self._run_drift()

        elapsed = time.perf_counter() - t0
        summary = self._build_summary(shap_result, fairness_result, drift_result)

        print(f"[MLens] Audit complete in {elapsed:.2f}s")

        return AuditReport(
            model_name=self.model_name,
            audit_timestamp=timestamp,
            runtime_seconds=elapsed,
            shap_result=shap_result,
            fairness_result=fairness_result,
            drift_result=drift_result,
            summary_lines=summary,
            metadata={
                "train_size": len(self.X_train),
                "test_size": len(self.X_test),
                "n_features": self.X_train.shape[1],
            },
        )

    # --------------------------------------------------------------- private

    def _run_shap(self) -> Optional[ShapResult]:
        if not self.run_shap:
            return None
        print("  [1/3] Computing SHAP values …")
        try:
            analyzer = ShapAnalyzer(
                model=self.model,
                background_data=self._sample(self.X_train, self.shap_background_samples),
                feature_names=self.feature_names,
            )
            return analyzer.explain(self.X_test)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"SHAP analysis failed: {exc}")
            return None

    def _run_fairness(self) -> Optional[FairnessResult]:
        if not self.run_fairness:
            return None
        print("  [2/3] Evaluating fairness metrics …")
        try:
            y_pred = self._predict(self.X_test)
            evaluator = FairnessEvaluator(
                y_true=self.y_test,
                y_pred=y_pred,
                sensitive_features=self.sensitive_features,
            )
            return evaluator.evaluate()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Fairness evaluation failed: {exc}")
            return None

    def _run_drift(self) -> Optional[DriftResult]:
        if not self.run_drift:
            return None
        print("  [3/3] Running drift detection …")
        try:
            detector = DriftDetector(
                reference=self.X_train,
                feature_names=self.feature_names,
            )
            return detector.detect(self.X_test)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Drift detection failed: {exc}")
            return None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions, handling sklearn and XGBoost models."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise TypeError(f"Model type '{type(self.model)}' does not expose .predict().")

    # ----------------------------------------------------------------- utils

    @staticmethod
    def _to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    @staticmethod
    def _infer_feature_names(X: Any) -> Optional[List[str]]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return None

    @staticmethod
    def _sample(X: np.ndarray, n: int) -> np.ndarray:
        if len(X) <= n:
            return X
        idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)
        return X[idx]

    @staticmethod
    def _build_summary(
        shap: Optional[ShapResult],
        fairness: Optional[FairnessResult],
        drift: Optional[DriftResult],
    ) -> List[str]:
        lines: List[str] = []

        if shap:
            top = shap.top_features(n=3)
            lines.append(
                f"Top 3 predictive features: {', '.join(f['name'] for f in top)}."
            )

        if fairness:
            if fairness.demographic_parity_gap > 0.1:
                lines.append(
                    f"⚠️  Demographic parity gap is {fairness.demographic_parity_gap:.3f} "
                    f"(threshold 0.10) — review for potential bias."
                )
            else:
                lines.append(
                    f"✅ Demographic parity gap ({fairness.demographic_parity_gap:.3f}) "
                    f"is within acceptable range."
                )

        if drift:
            drifted = [f for f in drift.feature_results if f["drifted"]]
            if drifted:
                names = ", ".join(f["feature"] for f in drifted[:5])
                lines.append(
                    f"⚠️  Data drift detected in {len(drifted)} feature(s): {names}."
                )
            else:
                lines.append("✅ No significant feature drift detected.")

        return lines
