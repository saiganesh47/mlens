"""
mlens/batch/batch_auditor.py
==============================
Audit many models from a directory or a list, producing
individual reports and a master summary CSV/HTML.

Supports:
  - Loading all .pkl / .joblib files from a folder
  - Running in parallel (ThreadPoolExecutor)
  - Generating a master summary table
  - Individual HTML report per model

Usage
-----
>>> from mlens.batch.batch_auditor import BatchAuditor
>>> batch = BatchAuditor(
...     model_dir="models/",
...     X_test=X_test,
...     y_test=y_test,
...     output_dir="batch_reports/",
... )
>>> summary = batch.run()
>>> summary.save_csv("batch_summary.csv")
>>> summary.save_html("batch_summary.html")
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class BatchAuditSummary:
    """
    Summary of a batch audit run.

    Attributes
    ----------
    n_models : int
        Number of models audited.
    n_success : int
        Models that audited without error.
    n_failed : int
        Models that raised an exception.
    summary_table : pd.DataFrame
        One row per model with key metrics.
    failed_models : dict
        {model_name: error_message} for any failures.
    runtime_seconds : float
        Total wall-clock runtime.
    output_dir : str
        Directory where per-model reports were saved.
    """

    n_models       : int
    n_success      : int
    n_failed       : int
    summary_table  : pd.DataFrame
    failed_models  : Dict[str, str]
    runtime_seconds: float
    output_dir     : str

    def save_csv(self, path: str = "batch_summary.csv") -> str:
        """Save summary table as CSV."""
        self.summary_table.to_csv(path)
        print(f"[MLens Batch] CSV saved → {Path(path).resolve()}")
        return str(Path(path).resolve())

    def save_html(self, path: str = "batch_summary.html") -> str:
        """Save an HTML summary table."""
        from mlens.batch.batch_report import BatchReportGenerator
        return BatchReportGenerator(self).render(path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_models":        self.n_models,
            "n_success":       self.n_success,
            "n_failed":        self.n_failed,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "failed_models":   self.failed_models,
            "summary":         self.summary_table.to_dict(orient="index"),
        }


# ── Batch Auditor ──────────────────────────────────────────────────────────

class BatchAuditor:
    """
    Audit many models in one call.

    Can be initialised with either:
      (a) A directory path containing .pkl / .joblib model files
      (b) A dict of {name: model} objects

    Parameters
    ----------
    X_test : array-like
        Test features shared across all models.
    y_test : array-like
        Test labels shared across all models.
    model_dir : str or Path, optional
        Directory of serialised model files.
    models : dict, optional
        Pre-loaded {name: model} dict.
    X_train : array-like, optional
        Training data for drift reference (defaults to X_test).
    sensitive_features : array-like, optional
        Protected attribute for fairness evaluation.
    feature_names : list of str, optional
        Column names for features.
    output_dir : str
        Directory to write per-model HTML reports (default: 'batch_reports').
    max_workers : int
        Thread pool size for parallel auditing (default: 4).
    run_shap : bool
    run_fairness : bool
    run_drift : bool
    shap_background_samples : int
    """

    def __init__(
        self,
        X_test                  : Any,
        y_test                  : Any,
        model_dir               : Optional[Union[str, Path]] = None,
        models                  : Optional[Dict[str, Any]]   = None,
        X_train                 : Optional[Any]              = None,
        sensitive_features      : Optional[Any]              = None,
        feature_names           : Optional[List[str]]        = None,
        output_dir              : str                        = "batch_reports",
        max_workers             : int                        = 4,
        run_shap                : bool                       = True,
        run_fairness            : bool                       = True,
        run_drift               : bool                       = True,
        shap_background_samples : int                        = 100,
    ) -> None:
        if model_dir is None and models is None:
            raise ValueError("Provide either model_dir or models.")

        self.X_test                  = X_test
        self.y_test                  = y_test
        self.X_train                 = X_train if X_train is not None else X_test
        self.sensitive_features      = sensitive_features
        self.feature_names           = feature_names
        self.output_dir              = Path(output_dir)
        self.max_workers             = max_workers
        self.run_shap                = run_shap
        self.run_fairness            = run_fairness and sensitive_features is not None
        self.run_drift               = run_drift
        self.shap_background_samples = shap_background_samples

        # Load models
        if models is not None:
            self._models = models
        else:
            self._models = self._load_from_dir(Path(model_dir))

    # ---------------------------------------------------------------- public

    def run(self) -> BatchAuditSummary:
        """
        Audit all models in parallel and return a BatchAuditSummary.

        Returns
        -------
        BatchAuditSummary
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        print(f"\n[MLens Batch] Auditing {len(self._models)} models "
              f"({self.max_workers} workers) …\n")

        rows         : List[Dict] = []
        failed_models: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._audit_one, name, model): name
                for name, model in self._models.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    row = future.result()
                    rows.append(row)
                    status = "✓" if not row.get("error") else "✗"
                    print(f"  {status} {name} ({row.get('runtime_s', '?')}s)")
                    if row.get("error"):
                        failed_models[name] = row["error"]
                except Exception as exc:
                    failed_models[name] = str(exc)
                    print(f"  ✗ {name} — {exc}")

        table     = pd.DataFrame(rows).set_index("model")
        elapsed   = time.perf_counter() - t0
        n_success = len(rows) - len(failed_models)

        print(f"\n[MLens Batch] Done — {n_success}/{len(self._models)} "
              f"succeeded in {elapsed:.2f}s")

        return BatchAuditSummary(
            n_models        = len(self._models),
            n_success       = n_success,
            n_failed        = len(failed_models),
            summary_table   = table,
            failed_models   = failed_models,
            runtime_seconds = elapsed,
            output_dir      = str(self.output_dir.resolve()),
        )

    # --------------------------------------------------------------- private

    def _audit_one(self, name: str, model: Any) -> Dict[str, Any]:
        """Audit a single model and return a flat metric row."""
        from mlens.auditor import ModelAuditor

        row: Dict[str, Any] = {"model": name, "error": None}

        try:
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
            report = auditor.run()

            # Save individual report
            report_path = self.output_dir / f"{name}_audit.html"
            report.save(report_path)
            row["report_path"] = str(report_path)

            # SHAP
            if report.shap_result:
                top = report.shap_result.top_features(n=1)
                row["top_feature"]      = top[0]["name"]          if top else None
                row["top_feature_shap"] = top[0]["mean_abs_shap"] if top else None

            # Fairness
            if report.fairness_result:
                fr = report.fairness_result
                row["dp_gap"]           = round(fr.demographic_parity_gap, 4)
                row["eo_gap"]           = round(fr.equalized_odds_gap,     4)
                row["disparate_impact"] = round(fr.disparate_impact,       4)
                row["fairness_flags"]   = len(fr.flags)
                row["is_fair"]          = fr.is_fair

            # Drift
            if report.drift_result:
                dr = report.drift_result
                psi = [f["psi"] for f in dr.feature_results]
                row["n_drifted"]    = dr.n_drifted
                row["max_psi"]      = round(max(psi), 4) if psi else 0.0
                row["drift_status"] = dr.overall_status

            row["runtime_s"] = round(report.runtime_seconds, 3)

        except Exception as exc:
            row["error"] = str(exc)

        return row

    @staticmethod
    def _load_from_dir(directory: Path) -> Dict[str, Any]:
        """Load all .pkl and .joblib model files from a directory."""
        import joblib, pickle

        models: Dict[str, Any] = {}
        for ext in ("*.pkl", "*.joblib"):
            for path in sorted(directory.glob(ext)):
                name = path.stem
                try:
                    models[name] = joblib.load(path)
                except Exception:
                    with open(path, "rb") as f:
                        models[name] = pickle.load(f)

        if not models:
            raise FileNotFoundError(
                f"No .pkl or .joblib files found in {directory}"
            )
        print(f"[MLens Batch] Loaded {len(models)} models from {directory}")
        return models
