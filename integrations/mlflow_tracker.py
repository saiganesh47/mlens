"""
mlens/integrations/mlflow_tracker.py
======================================
Log MLens audit results to MLflow as a run with metrics,
parameters, and a downloadable HTML report artifact.

Usage
-----
>>> from mlens.integrations.mlflow_tracker import MLflowTracker
>>> tracker = MLflowTracker(experiment_name="mlens-audits")
>>> tracker.log(report)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional


class MLflowTracker:
    """
    Logs an AuditReport to MLflow.

    Logs the following:
      - **Params:**  model_name, audit_timestamp, n_features, test_size
      - **Metrics:** dp_gap, eo_gap, disparate_impact, n_drifted,
                     max_psi, runtime_seconds
      - **Tags:**    drift_status, fairness_status
      - **Artifact:** Full HTML audit report

    Parameters
    ----------
    experiment_name : str
        MLflow experiment to log under (default: 'mlens-audits').
    tracking_uri : str, optional
        MLflow tracking server URI. If None, uses local ./mlruns.
    run_name : str, optional
        Override the MLflow run name.
    """

    def __init__(
        self,
        experiment_name: str = "mlens-audits",
        tracking_uri   : Optional[str] = None,
        run_name       : Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri    = tracking_uri
        self.run_name        = run_name

    # ---------------------------------------------------------------- public

    def log(self, report) -> str:
        """
        Log an AuditReport to MLflow and return the run_id.

        Parameters
        ----------
        report : AuditReport
            Populated report from ModelAuditor.run().

        Returns
        -------
        str : MLflow run_id
        """
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLflow is required. Install with: pip install mlflow"
            )

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        run_name = self.run_name or f"audit-{report.model_name}-{report.audit_timestamp}"

        with mlflow.start_run(run_name=run_name) as run:

            # ── Parameters ────────────────────────────────────────────────
            mlflow.log_params({
                "model_name":       report.model_name,
                "audit_timestamp":  report.audit_timestamp,
                "n_features":       report.metadata.get("n_features", 0),
                "train_size":       report.metadata.get("train_size", 0),
                "test_size":        report.metadata.get("test_size", 0),
            })

            # ── Metrics ───────────────────────────────────────────────────
            metrics: dict = {
                "runtime_seconds": report.runtime_seconds,
            }

            if report.fairness_result:
                fr = report.fairness_result
                metrics.update({
                    "fairness.demographic_parity_gap": fr.demographic_parity_gap,
                    "fairness.equalized_odds_gap":     fr.equalized_odds_gap,
                    "fairness.disparate_impact":       fr.disparate_impact,
                    "fairness.n_flags":                len(fr.flags),
                })

            if report.drift_result:
                dr = report.drift_result
                psi_vals = [f["psi"] for f in dr.feature_results]
                metrics.update({
                    "drift.n_drifted":  dr.n_drifted,
                    "drift.max_psi":    max(psi_vals) if psi_vals else 0.0,
                    "drift.mean_psi":   sum(psi_vals) / len(psi_vals) if psi_vals else 0.0,
                })

            if report.shap_result:
                top = report.shap_result.top_features(n=1)
                if top:
                    metrics["shap.top_feature_importance"] = top[0]["mean_abs_shap"]

            mlflow.log_metrics(metrics)

            # ── Tags ──────────────────────────────────────────────────────
            tags: dict = {"mlens.version": "0.4.0"}

            if report.drift_result:
                tags["drift.overall_status"] = report.drift_result.overall_status

            if report.fairness_result:
                tags["fairness.is_fair"] = str(report.fairness_result.is_fair)

            mlflow.set_tags(tags)

            # ── Artifact: HTML report ─────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmp_dir:
                html_path = Path(tmp_dir) / f"mlens_audit_{report.model_name}.html"
                report.save(html_path)
                mlflow.log_artifact(str(html_path), artifact_path="mlens_report")

            # ── Summary as note ───────────────────────────────────────────
            note = "\n".join(report.summary_lines)
            mlflow.set_tag("mlflow.note.content", note)

            run_id = run.info.run_id
            print(f"[MLens → MLflow] Run logged: {run_id}")
            print(f"[MLens → MLflow] Experiment: {self.experiment_name}")
            return run_id
