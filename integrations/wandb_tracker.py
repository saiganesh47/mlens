"""
mlens/integrations/wandb_tracker.py
=====================================
Log MLens audit results to Weights & Biases (W&B).

Logs metrics, config, a summary table, and the HTML report
as a W&B Artifact for versioned model auditing.

Usage
-----
>>> from mlens.integrations.wandb_tracker import WandbTracker
>>> tracker = WandbTracker(project="mlens-audits")
>>> tracker.log(report)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional


class WandbTracker:
    """
    Logs an AuditReport to Weights & Biases.

    Logs the following:
      - **Config:**   model_name, timestamp, metadata
      - **Metrics:**  fairness gaps, drift PSI, runtime
      - **Table:**    per-group fairness breakdown
      - **Artifact:** Full HTML audit report (versioned)

    Parameters
    ----------
    project : str
        W&B project name (default: 'mlens-audits').
    entity : str, optional
        W&B username or team. If None, uses default entity.
    run_name : str, optional
        Override the W&B run name.
    tags : list of str, optional
        W&B tags for this run.
    """

    def __init__(
        self,
        project  : str = "mlens-audits",
        entity   : Optional[str] = None,
        run_name : Optional[str] = None,
        tags     : Optional[list] = None,
    ) -> None:
        self.project  = project
        self.entity   = entity
        self.run_name = run_name
        self.tags     = tags or ["mlens", "audit"]

    # ---------------------------------------------------------------- public

    def log(self, report) -> str:
        """
        Log an AuditReport to W&B and return the run URL.

        Parameters
        ----------
        report : AuditReport

        Returns
        -------
        str : W&B run URL
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "W&B is required. Install with: pip install wandb"
            )

        run_name = self.run_name or f"audit-{report.model_name}"

        run = wandb.init(
            project = self.project,
            entity  = self.entity,
            name    = run_name,
            tags    = self.tags,
            config  = {
                "model_name":      report.model_name,
                "audit_timestamp": report.audit_timestamp,
                "n_features":      report.metadata.get("n_features", 0),
                "train_size":      report.metadata.get("train_size", 0),
                "test_size":       report.metadata.get("test_size", 0),
                "mlens_version":   "0.4.0",
            },
        )

        # ── Metrics ───────────────────────────────────────────────────────
        metrics: dict = {"runtime_seconds": report.runtime_seconds}

        if report.fairness_result:
            fr = report.fairness_result
            metrics.update({
                "fairness/demographic_parity_gap": fr.demographic_parity_gap,
                "fairness/equalized_odds_gap":     fr.equalized_odds_gap,
                "fairness/disparate_impact":       fr.disparate_impact,
                "fairness/n_flags":                len(fr.flags),
            })

        if report.drift_result:
            dr = report.drift_result
            psi_vals = [f["psi"] for f in dr.feature_results]
            metrics.update({
                "drift/n_drifted":  dr.n_drifted,
                "drift/max_psi":    max(psi_vals) if psi_vals else 0.0,
                "drift/mean_psi":   sum(psi_vals) / len(psi_vals) if psi_vals else 0.0,
            })

        if report.shap_result:
            top = report.shap_result.top_features(n=3)
            for feat in top:
                safe_name = feat["name"].replace(" ", "_")
                metrics[f"shap/feature_{feat['rank']}_{safe_name}"] = feat["mean_abs_shap"]

        wandb.log(metrics)

        # ── Fairness table ────────────────────────────────────────────────
        if report.fairness_result and report.fairness_result.per_group_metrics:
            import wandb as wb
            rows = report.fairness_result.per_group_metrics
            if rows:
                cols = list(rows[0].keys())
                table = wb.Table(
                    columns=cols,
                    data=[[row.get(c, "") for c in cols] for row in rows],
                )
                wandb.log({"fairness/per_group_table": table})

        # ── SHAP bar chart ────────────────────────────────────────────────
        if report.shap_result:
            import wandb as wb
            top = report.shap_result.top_features(n=10)
            shap_table = wb.Table(
                columns=["rank", "feature", "mean_abs_shap"],
                data=[[f["rank"], f["name"], f["mean_abs_shap"]] for f in top],
            )
            wandb.log({"shap/top_features": shap_table})

        # ── HTML report as artifact ───────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"mlens_audit_{report.model_name}.html"
            report.save(html_path)
            artifact = wandb.Artifact(
                name        = f"mlens-audit-{report.model_name}",
                type        = "audit-report",
                description = f"MLens audit for {report.model_name} at {report.audit_timestamp}",
                metadata    = report.to_dict(),
            )
            artifact.add_file(str(html_path))
            run.log_artifact(artifact)

        # ── Summary ───────────────────────────────────────────────────────
        wandb.summary["audit_summary"] = "\n".join(report.summary_lines)

        run_url = run.get_url()
        print(f"[MLens → W&B] Run logged: {run_url}")
        wandb.finish()
        return run_url
