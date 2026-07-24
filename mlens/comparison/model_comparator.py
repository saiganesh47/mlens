"""
mlens/comparison/model_comparator.py
======================================
Compare multiple trained models side-by-side across
explainability, fairness, and drift dimensions.

Usage
-----
>>> from mlens.comparison.model_comparator import ModelComparator
>>> comparator = ModelComparator(
...     models={"RF": rf_model, "GBT": gbt_model, "XGB": xgb_model},
...     X_train=X_train, X_test=X_test, y_test=y_test,
...     sensitive_features=s_test,
... )
>>> result = comparator.compare()
>>> result.best_model()
>>> result.save("comparison_report.html")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """
    Side-by-side audit results for N models.

    Attributes
    ----------
    model_names : list of str
        Names of the models compared.
    audit_reports : dict
        Raw AuditReport per model name.
    comparison_table : pd.DataFrame
        One row per model, columns = key metrics.
    timestamp : str
        UTC timestamp of the comparison run.
    runtime_seconds : float
        Total runtime across all model audits.
    """

    model_names     : List[str]
    audit_reports   : Dict[str, Any]
    comparison_table: pd.DataFrame
    timestamp       : str
    runtime_seconds : float

    # ---------------------------------------------------------------- public

    def best_model(
        self,
        metric : str = "accuracy",
        higher_is_better: bool = True,
    ) -> Tuple[str, float]:
        """
        Return the best model name and its score on the given metric.

        Parameters
        ----------
        metric : str
            Column name in comparison_table (default: 'accuracy').
        higher_is_better : bool
            True for accuracy/F1; False for dp_gap/drift_psi.

        Returns
        -------
        Tuple[str, float] : (model_name, score)
        """
        if metric not in self.comparison_table.columns:
            raise ValueError(
                f"Metric '{metric}' not in comparison table. "
                f"Available: {list(self.comparison_table.columns)}"
            )
        col = self.comparison_table[metric].dropna()
        idx = col.idxmax() if higher_is_better else col.idxmin()
        return idx, float(col[idx])

    def rank(self, metric: str = "accuracy", higher_is_better: bool = True) -> pd.DataFrame:
        """Return comparison_table sorted by the given metric."""
        asc = not higher_is_better
        return self.comparison_table.sort_values(metric, ascending=asc)

    def save(self, path: str = "comparison_report.html") -> str:
        """Save an HTML comparison report and return the output path."""
        from mlens.comparison.comparison_report import ComparisonReportGenerator
        generator = ComparisonReportGenerator(self)
        return generator.render(path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_names":    self.model_names,
            "timestamp":      self.timestamp,
            "runtime":        round(self.runtime_seconds, 3),
            "comparison":     self.comparison_table.to_dict(orient="index"),
        }


# ── Comparator ─────────────────────────────────────────────────────────────

class ModelComparator:
    """
    Audit N models with the same data and produce a side-by-side comparison.

    Parameters
    ----------
    models : dict of {name: model}
        Trained models to compare. Values must be sklearn-compatible.
    X_train : array-like
        Training features (drift reference + SHAP background).
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    sensitive_features : array-like, optional
        Protected attribute for fairness evaluation.
    feature_names : list of str, optional
        Column names inferred from DataFrames if not supplied.
    shap_background_samples : int
        Background rows for SHAP explainer (default: 100).
    run_shap : bool
    run_fairness : bool
    run_drift : bool
    """

    def __init__(
        self,
        models                  : Dict[str, Any],
        X_train                 : Any,
        X_test                  : Any,
        y_test                  : Any,
        sensitive_features      : Optional[Any] = None,
        feature_names           : Optional[List[str]] = None,
        shap_background_samples : int = 100,
        run_shap                : bool = True,
        run_fairness            : bool = True,
        run_drift               : bool = True,
    ) -> None:
        self.models                   = models
        self.X_train                  = X_train
        self.X_test                   = X_test
        self.y_test                   = y_test
        self.sensitive_features       = sensitive_features
        self.feature_names            = feature_names
        self.shap_background_samples  = shap_background_samples
        self.run_shap                 = run_shap
        self.run_fairness             = run_fairness and sensitive_features is not None
        self.run_drift                = run_drift

    # ---------------------------------------------------------------- public

    def compare(self) -> ComparisonResult:
        """
        Audit all models and return a ComparisonResult.

        The pipeline runs ModelAuditor on each model sequentially
        and aggregates key metrics into a comparison DataFrame.

        Returns
        -------
        ComparisonResult
        """
        from mlens.auditor import ModelAuditor

        t0        = time.perf_counter()
        timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        reports   : Dict[str, Any] = {}
        rows      : List[Dict]     = []

        print(f"\n[MLens Comparator] Comparing {len(self.models)} models …\n")

        for name, model in self.models.items():
            print(f"  → Auditing '{name}' …")
            auditor = ModelAuditor(
                model                    = model,
                X_train                  = self.X_train,
                X_test                   = self.X_test,
                y_test                   = self.y_test,
                sensitive_features       = self.sensitive_features,
                feature_names            = self.feature_names,
                model_name               = name,
                shap_background_samples  = self.shap_background_samples,
                run_shap                 = self.run_shap,
                run_fairness             = self.run_fairness,
                run_drift                = self.run_drift,
            )
            report        = auditor.run()
            reports[name] = report
            rows.append(self._extract_row(name, report))
            print(f"     ✓ done ({report.runtime_seconds:.2f}s)")

        table   = pd.DataFrame(rows).set_index("model")
        elapsed = time.perf_counter() - t0

        print(f"\n[MLens Comparator] All models audited in {elapsed:.2f}s\n")
        print(table.to_string())

        return ComparisonResult(
            model_names      = list(self.models.keys()),
            audit_reports    = reports,
            comparison_table = table,
            timestamp        = timestamp,
            runtime_seconds  = elapsed,
        )

    # --------------------------------------------------------------- private

    def _extract_row(self, name: str, report: Any) -> Dict[str, Any]:
        """Extract a flat row of key metrics from an AuditReport."""
        row: Dict[str, Any] = {"model": name}

        # ── Accuracy
        try:
            y_pred = report._auditor_model_ref.predict(
                np.array(self.X_test)
            ) if hasattr(report, "_auditor_model_ref") else None
        except Exception:
            y_pred = None

        # ── SHAP top feature importance
        if report.shap_result:
            top = report.shap_result.top_features(n=1)
            row["top_feature"]       = top[0]["name"]          if top else None
            row["top_feature_shap"]  = top[0]["mean_abs_shap"] if top else None

        # ── Fairness
        if report.fairness_result:
            fr = report.fairness_result
            row["dp_gap"]          = round(fr.demographic_parity_gap, 4)
            row["eo_gap"]          = round(fr.equalized_odds_gap,     4)
            row["disparate_impact"]= round(fr.disparate_impact,       4)
            row["fairness_flags"]  = len(fr.flags)
            row["is_fair"]         = fr.is_fair

        # ── Drift
        if report.drift_result:
            dr        = report.drift_result
            psi_vals  = [f["psi"] for f in dr.feature_results]
            row["n_drifted"]       = dr.n_drifted
            row["max_psi"]         = round(max(psi_vals), 4) if psi_vals else 0.0
            row["drift_status"]    = dr.overall_status

        # ── Runtime
        row["runtime_s"] = round(report.runtime_seconds, 3)

        return row
