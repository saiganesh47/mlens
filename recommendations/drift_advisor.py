"""
mlens/recommendations/drift_advisor.py
========================================
Analyses drift results and generates a plain-English retraining
schedule and remediation plan.

Usage
-----
>>> from mlens.recommendations.drift_advisor import DriftAdvisor
>>> advisor = DriftAdvisor(report)
>>> plan = advisor.advise()
>>> print(plan.retraining_urgency)
>>> for action in plan.actions:
...     print(action.title)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Action and Plan dataclasses ────────────────────────────────────────────

@dataclass
class DriftAction:
    priority        : str
    title           : str
    explanation     : str
    code_snippet    : str
    expected_impact : str


@dataclass
class DriftPlan:
    """
    Drift remediation plan from DriftAdvisor.

    Attributes
    ----------
    model_name : str
    overall_status : str        'stable' | 'moderate' | 'significant'
    retraining_urgency : str    'immediate' | 'scheduled' | 'monitor'
    drifted_features : list     Features with detected drift.
    actions : list of DriftAction
    retraining_schedule : str   Plain-English retraining recommendation.
    summary : str
    """
    model_name           : str
    overall_status       : str
    retraining_urgency   : str
    drifted_features     : List[str]
    actions              : List[DriftAction]
    retraining_schedule  : str
    summary              : str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":          self.model_name,
            "overall_status":      self.overall_status,
            "retraining_urgency":  self.retraining_urgency,
            "drifted_features":    self.drifted_features,
            "retraining_schedule": self.retraining_schedule,
            "summary":             self.summary,
            "n_actions":           len(self.actions),
        }


# ── Advisor ────────────────────────────────────────────────────────────────

class DriftAdvisor:
    """
    Generate a drift remediation and retraining plan from an AuditReport.

    Parameters
    ----------
    report : AuditReport
        Populated report from ModelAuditor.run().
    """

    def __init__(self, report: Any) -> None:
        self.report = report

    # ---------------------------------------------------------------- public

    def advise(self) -> DriftPlan:
        """
        Analyse drift results and return a DriftPlan.

        Returns
        -------
        DriftPlan
        """
        dr = self.report.drift_result
        if dr is None:
            return DriftPlan(
                model_name          = self.report.model_name,
                overall_status      = "unknown",
                retraining_urgency  = "monitor",
                drifted_features    = [],
                actions             = [],
                retraining_schedule = "Drift detection was not run.",
                summary             = "Enable drift detection by passing X_train to ModelAuditor.",
            )

        drifted  = dr.drifted_features()
        urgency  = self._urgency(dr)
        schedule = self._retraining_schedule(dr, urgency)
        actions  = self._build_actions(dr, drifted, urgency)
        summary  = self._build_summary(dr, drifted, urgency)

        return DriftPlan(
            model_name          = self.report.model_name,
            overall_status      = dr.overall_status,
            retraining_urgency  = urgency,
            drifted_features    = drifted,
            actions             = actions,
            retraining_schedule = schedule,
            summary             = summary,
        )

    # --------------------------------------------------------------- private

    def _urgency(self, dr: Any) -> str:
        if dr.overall_status == "significant" or dr.n_drifted >= 3:
            return "immediate"
        if dr.overall_status == "moderate" or dr.n_drifted >= 1:
            return "scheduled"
        return "monitor"

    def _retraining_schedule(self, dr: Any, urgency: str) -> str:
        schedules = {
            "immediate": (
                "🔴 Retrain immediately. Significant drift in "
                f"{dr.n_drifted} feature(s) likely degrades model performance. "
                "Deploy a retrained model within 24-48 hours."
            ),
            "scheduled": (
                "🟡 Schedule a retraining run within the next 1-2 weeks. "
                "Moderate drift detected — model is likely still usable "
                "but degrading. Monitor error rates closely."
            ),
            "monitor": (
                "🟢 No immediate retraining needed. "
                "Continue monitoring weekly. Schedule a retraining "
                "review in 30 days or when PSI exceeds 0.10."
            ),
        }
        return schedules.get(urgency, "Monitor the model.")

    def _build_actions(
        self, dr: Any, drifted: List[str], urgency: str
    ) -> List[DriftAction]:
        actions: List[DriftAction] = []

        # ── Retrain on fresh data ──────────────────────────────────────────
        actions.append(DriftAction(
            priority        = "high" if urgency == "immediate" else "medium",
            title           = "Retrain on recent data window",
            explanation     = (
                "The simplest and most effective fix — retrain the model "
                "on data collected in the past 30-90 days to reflect "
                "the new distribution."
            ),
            code_snippet    = """# Assuming you have a date column in your dataframe
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=90)
recent_data = df[df["date"] >= cutoff]

X_recent = recent_data.drop(columns=["label", "date"])
y_recent = recent_data["label"]

model.fit(X_recent, y_recent)
""",
            expected_impact = "Realigns model to current distribution",
        ))

        # ── Drop drifted features ──────────────────────────────────────────
        if drifted:
            feat_str = '", "'.join(drifted[:3])
            actions.append(DriftAction(
                priority        = "medium",
                title           = f"Remove or transform drifted features: {', '.join(drifted[:3])}",
                explanation     = (
                    "If drifted features are not critical predictors, "
                    "removing them stabilises the model against future drift."
                ),
                code_snippet    = f"""# Drop drifted features before retraining
drifted_features = ["{feat_str}"]
X_train_stable = X_train.drop(columns=drifted_features, errors="ignore")
X_test_stable  = X_test.drop(columns=drifted_features,  errors="ignore")

model.fit(X_train_stable, y_train)
""",
                expected_impact = "Stable predictions on future data",
            ))

        # ── Rolling window retraining ──────────────────────────────────────
        actions.append(DriftAction(
            priority        = "medium",
            title           = "Implement rolling window retraining",
            explanation     = (
                "Automate periodic retraining on a sliding window of "
                "recent data. Prevents drift accumulation over time."
            ),
            code_snippet    = """# Schedule with Airflow, Prefect, or a cron job
from mlens import ModelAuditor

def weekly_retrain(model, X_new, y_new, X_test, y_test, sensitive_test):
    model.fit(X_new, y_new)

    # Audit the retrained model
    report = ModelAuditor(model, X_new, X_test, y_test,
                           sensitive_features=sensitive_test).run()

    # Alert if still drifted
    from mlens.monitoring import AlertManager
    AlertManager(slack_webhook="...").check_and_notify(report)
    return model, report
""",
            expected_impact = "Prevents drift accumulation",
        ))

        # ── Concept drift monitoring ───────────────────────────────────────
        actions.append(DriftAction(
            priority        = "low",
            title           = "Enable ADWIN concept drift monitoring",
            explanation     = (
                "Track model error rate in real-time with ADWIN. "
                "Alerts when prediction accuracy degrades — catches "
                "concept drift before PSI does."
            ),
            code_snippet    = """from mlens.drift.concept_drift import ConceptDriftDetector

detector = ConceptDriftDetector(method="adwin")

# In your prediction pipeline, after each batch:
result = detector.detect(y_true_batch, y_pred_batch)
if result.drift_detected:
    print(result.summary)
    # trigger retraining pipeline
""",
            expected_impact = "Early warning before performance degrades",
        ))

        return actions

    @staticmethod
    def _build_summary(dr: Any, drifted: List[str], urgency: str) -> str:
        emoji = {"immediate": "🔴", "scheduled": "🟡", "monitor": "🟢"}.get(urgency, "⚪")
        return (
            f"{emoji} Overall drift status: {dr.overall_status.upper()}. "
            f"{dr.n_drifted} of {len(dr.feature_results)} features drifted "
            f"({', '.join(drifted[:3])}{'…' if len(drifted) > 3 else ''}). "
            f"Retraining urgency: {urgency.upper()}."
        )
