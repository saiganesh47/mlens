"""
mlens/recommendations/fairness_advisor.py
==========================================
Analyses fairness violations in an AuditReport and generates
a prioritised, plain-English action plan for remediation.

Usage
-----
>>> from mlens.recommendations.fairness_advisor import FairnessAdvisor
>>> advisor = FairnessAdvisor(report)
>>> plan = advisor.advise()
>>> for action in plan.actions:
...     print(action.priority, action.title)
...     print(action.code_snippet)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Action dataclass ───────────────────────────────────────────────────────

@dataclass
class FairnessAction:
    """
    A single remediation action.

    Attributes
    ----------
    priority : str      'critical' | 'high' | 'medium' | 'low'
    category : str      'data' | 'model' | 'post-processing' | 'monitoring'
    title : str         Short action title.
    explanation : str   Why this action helps.
    code_snippet : str  Runnable Python code implementing the fix.
    expected_impact : str  What metric should improve.
    """
    priority        : str
    category        : str
    title           : str
    explanation     : str
    code_snippet    : str
    expected_impact : str


@dataclass
class FairnessPlan:
    """
    Complete remediation plan from FairnessAdvisor.

    Attributes
    ----------
    model_name : str
    violations : list of str   Triggered fairness flags.
    actions : list of FairnessAction   Sorted by priority.
    summary : str   One-paragraph executive summary.
    """
    model_name : str
    violations : List[str]
    actions    : List[FairnessAction]
    summary    : str

    def critical(self) -> List[FairnessAction]:
        return [a for a in self.actions if a.priority == "critical"]

    def by_category(self, category: str) -> List[FairnessAction]:
        return [a for a in self.actions if a.category == category]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "violations": self.violations,
            "summary":    self.summary,
            "actions": [
                {
                    "priority":        a.priority,
                    "category":        a.category,
                    "title":           a.title,
                    "explanation":     a.explanation,
                    "expected_impact": a.expected_impact,
                }
                for a in self.actions
            ],
        }


# ── Advisor ────────────────────────────────────────────────────────────────

class FairnessAdvisor:
    """
    Generate a prioritised fairness remediation plan from an AuditReport.

    Parameters
    ----------
    report : AuditReport
        Populated report from ModelAuditor.run().
    dp_critical : float
        DP gap above which actions are marked 'critical' (default: 0.20).
    di_critical : float
        Disparate impact below which actions are marked 'critical' (default: 0.70).
    """

    def __init__(
        self,
        report      : Any,
        dp_critical : float = 0.20,
        di_critical : float = 0.70,
    ) -> None:
        self.report      = report
        self.dp_critical = dp_critical
        self.di_critical = di_critical

    # ---------------------------------------------------------------- public

    def advise(self) -> FairnessPlan:
        """
        Analyse the report's fairness result and return a FairnessPlan.

        Returns
        -------
        FairnessPlan
        """
        fr = self.report.fairness_result
        if fr is None:
            return FairnessPlan(
                model_name = self.report.model_name,
                violations = [],
                actions    = [],
                summary    = "No fairness evaluation was run. "
                             "Pass sensitive_features to ModelAuditor to enable it.",
            )

        violations = fr.flags
        actions    : List[FairnessAction] = []

        # ── Data-level fixes ───────────────────────────────────────────────
        if fr.demographic_parity_gap > 0.10:
            actions.append(self._reweight_action(fr))
            actions.append(self._resample_action(fr))

        # ── Model-level fixes ──────────────────────────────────────────────
        if fr.demographic_parity_gap > 0.10:
            actions.append(self._fairlearn_constraint_action(fr))
            actions.append(self._balanced_class_weight_action(fr))

        if fr.equalized_odds_gap > 0.10:
            actions.append(self._equalized_odds_action(fr))

        # ── Post-processing fixes ──────────────────────────────────────────
        if fr.disparate_impact < 0.80:
            actions.append(self._threshold_optimizer_action(fr))

        # ── Monitoring ────────────────────────────────────────────────────
        actions.append(self._monitoring_action())

        # Promote critical priority
        for a in actions:
            if (fr.demographic_parity_gap > self.dp_critical or
                    fr.disparate_impact < self.di_critical):
                if a.priority == "high":
                    a.priority = "critical"

        # Sort by priority
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(key=lambda a: order.get(a.priority, 4))

        summary = self._build_summary(fr, actions)

        return FairnessPlan(
            model_name = self.report.model_name,
            violations = violations,
            actions    = actions,
            summary    = summary,
        )

    # ── Action builders ────────────────────────────────────────────────────

    def _reweight_action(self, fr) -> FairnessAction:
        feat = fr.sensitive_feature_name
        return FairnessAction(
            priority        = "high",
            category        = "data",
            title           = "Re-weight training samples by group",
            explanation     = (
                f"The demographic parity gap ({fr.demographic_parity_gap:.3f}) "
                f"suggests '{feat}' groups are under/over-represented. "
                "Sample weights can compensate without changing the model."
            ),
            code_snippet    = f"""from sklearn.utils.class_weight import compute_sample_weight

# Compute per-sample weights based on sensitive group membership
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=sensitive_train,   # use the sensitive feature as the class
)

# Pass to model.fit()
model.fit(X_train, y_train, sample_weight=sample_weights)
""",
            expected_impact = "Reduced demographic parity gap by 20-40%",
        )

    def _resample_action(self, fr) -> FairnessAction:
        return FairnessAction(
            priority        = "medium",
            category        = "data",
            title           = "Oversample underrepresented group (SMOTE)",
            explanation     = (
                "Oversampling the minority group in the training data "
                "can reduce selection rate disparities."
            ),
            code_snippet    = """from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
model.fit(X_resampled, y_resampled)
""",
            expected_impact = "Improved recall parity across groups",
        )

    def _fairlearn_constraint_action(self, fr) -> FairnessAction:
        feat = fr.sensitive_feature_name
        return FairnessAction(
            priority        = "high",
            category        = "model",
            title           = "Fairness-constrained training (Fairlearn)",
            explanation     = (
                "ExponentiatedGradient directly minimises demographic "
                "parity gap as a training constraint — the most reliable "
                "method for closing disparity gaps."
            ),
            code_snippet    = f"""from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import GradientBoostingClassifier

estimator  = GradientBoostingClassifier(random_state=42)
constraint = DemographicParity()

mitigator  = ExponentiatedGradient(estimator, constraint)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

# Evaluate
y_pred = mitigator.predict(X_test)
""",
            expected_impact = "Enforced fairness constraint during training",
        )

    def _balanced_class_weight_action(self, fr) -> FairnessAction:
        return FairnessAction(
            priority        = "medium",
            category        = "model",
            title           = "Use class_weight='balanced'",
            explanation     = (
                "Setting class_weight='balanced' adjusts the loss function "
                "to account for group imbalances with no architecture change."
            ),
            code_snippet    = """from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # ← add this
    random_state=42,
)
model.fit(X_train, y_train)
""",
            expected_impact = "Improved recall for minority group",
        )

    def _equalized_odds_action(self, fr) -> FairnessAction:
        return FairnessAction(
            priority        = "high",
            category        = "model",
            title           = "Equalized Odds constraint (Fairlearn)",
            explanation     = (
                f"Equalized odds gap is {fr.equalized_odds_gap:.3f}. "
                "This constraint ensures equal TPR and FPR across groups."
            ),
            code_snippet    = """from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

mitigator = ExponentiatedGradient(
    estimator=GradientBoostingClassifier(random_state=42),
    constraints=EqualizedOdds(),
)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
""",
            expected_impact = "Equalised TPR/FPR across groups",
        )

    def _threshold_optimizer_action(self, fr) -> FairnessAction:
        return FairnessAction(
            priority        = "high",
            category        = "post-processing",
            title           = "Per-group threshold optimisation",
            explanation     = (
                f"Disparate impact is {fr.disparate_impact:.3f} (below EEOC 0.80). "
                "Setting different decision thresholds per group is a fast, "
                "model-agnostic fix."
            ),
            code_snippet    = """from fairlearn.postprocessing import ThresholdOptimizer

optimizer = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",
    predict_method="predict_proba",
    objective="balanced_accuracy_score",
)
optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
y_pred = optimizer.predict(X_test, sensitive_features=sensitive_test)
""",
            expected_impact = "Disparate impact raised above 0.80",
        )

    def _monitoring_action(self) -> FairnessAction:
        return FairnessAction(
            priority        = "medium",
            category        = "monitoring",
            title           = "Schedule regular fairness audits",
            explanation     = (
                "Fairness metrics can degrade over time as data distributions "
                "shift. Automated scheduled audits catch regressions early."
            ),
            code_snippet    = """from mlens import ModelAuditor
from mlens.monitoring import AlertManager

# Run weekly in a cron job / Airflow DAG
auditor = ModelAuditor(model, X_train, X_test, y_test,
                        sensitive_features=sensitive_test)
report  = auditor.run()

# Alert if fairness degrades
alert = AlertManager(slack_webhook="https://hooks.slack.com/...")
alert.check_and_notify(report)
""",
            expected_impact = "Early warning on fairness degradation",
        )

    @staticmethod
    def _build_summary(fr: Any, actions: List[FairnessAction]) -> str:
        critical_n = sum(1 for a in actions if a.priority == "critical")
        high_n     = sum(1 for a in actions if a.priority == "high")
        lines = [
            f"Demographic parity gap: {fr.demographic_parity_gap:.4f} "
            f"({'⚠️ critical' if fr.demographic_parity_gap > 0.20 else '⚠️ flagged'}) — "
            f"threshold 0.10.",
            f"Disparate impact: {fr.disparate_impact:.4f} "
            f"({'⚠️ below EEOC 4/5ths' if fr.disparate_impact < 0.80 else '✅ OK'}).",
            f"{len(actions)} remediation action(s) identified "
            f"({critical_n} critical, {high_n} high priority).",
            "Recommended first step: " + (actions[0].title if actions else "N/A") + ".",
        ]
        return " ".join(lines)
