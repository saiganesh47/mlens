"""
tests/test_advisors.py
========================
Unit tests for FairnessAdvisor, DriftAdvisor, and ShapAdvisor.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from mlens.recommendations.fairness_advisor import FairnessAdvisor, FairnessPlan
from mlens.recommendations.drift_advisor    import DriftAdvisor, DriftPlan
from mlens.recommendations.shap_advisor     import ShapAdvisor, ShapPlan


# ── Shared fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_report():
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    sensitive = np.random.choice(["A", "B"], size=len(y_test), replace=True)
    model = GradientBoostingClassifier(n_estimators=20, random_state=0)
    model.fit(X_train, y_train)

    from mlens.auditor import ModelAuditor
    return ModelAuditor(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        sensitive_features=sensitive,
        model_name="TestGBT",
        shap_background_samples=30,
    ).run()


# ── FairnessAdvisor tests ──────────────────────────────────────────────────

class TestFairnessAdvisor:

    def test_returns_fairness_plan(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        assert isinstance(plan, FairnessPlan)

    def test_plan_has_model_name(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        assert plan.model_name == "TestGBT"

    def test_plan_has_actions(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        assert isinstance(plan.actions, list)

    def test_plan_has_summary(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        assert len(plan.summary) > 0

    def test_action_priorities_valid(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        valid = {"critical", "high", "medium", "low"}
        for a in plan.actions:
            assert a.priority in valid

    def test_action_categories_valid(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        valid = {"data", "model", "post-processing", "monitoring"}
        for a in plan.actions:
            assert a.category in valid

    def test_action_has_code_snippet(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        for a in plan.actions:
            assert len(a.code_snippet) > 0

    def test_critical_method(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        critical = plan.critical()
        assert all(a.priority == "critical" for a in critical)

    def test_by_category(self, full_report):
        plan = FairnessAdvisor(full_report).advise()
        data_actions = plan.by_category("data")
        assert all(a.category == "data" for a in data_actions)

    def test_to_dict_keys(self, full_report):
        d = FairnessAdvisor(full_report).advise().to_dict()
        assert "model_name" in d
        assert "violations" in d
        assert "actions"    in d
        assert "summary"    in d

    def test_no_fairness_result_returns_plan(self):
        """Should return a plan even with no fairness result."""
        class MockReport:
            model_name     = "Mock"
            fairness_result= None
            drift_result   = None
            shap_result    = None
        plan = FairnessAdvisor(MockReport()).advise()
        assert isinstance(plan, FairnessPlan)
        assert len(plan.actions) == 0


# ── DriftAdvisor tests ─────────────────────────────────────────────────────

class TestDriftAdvisor:

    def test_returns_drift_plan(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert isinstance(plan, DriftPlan)

    def test_plan_has_model_name(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert plan.model_name == "TestGBT"

    def test_overall_status_valid(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert plan.overall_status in ("stable", "moderate", "significant")

    def test_retraining_urgency_valid(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert plan.retraining_urgency in ("immediate", "scheduled", "monitor")

    def test_retraining_schedule_not_empty(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert len(plan.retraining_schedule) > 0

    def test_actions_not_empty(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert len(plan.actions) > 0

    def test_actions_have_code(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        for a in plan.actions:
            assert len(a.code_snippet) > 0

    def test_summary_not_empty(self, full_report):
        plan = DriftAdvisor(full_report).advise()
        assert len(plan.summary) > 0

    def test_to_dict_keys(self, full_report):
        d = DriftAdvisor(full_report).advise().to_dict()
        assert "overall_status"     in d
        assert "retraining_urgency" in d
        assert "drifted_features"   in d
        assert "summary"            in d

    def test_no_drift_result_returns_plan(self):
        class MockReport:
            model_name     = "Mock"
            fairness_result= None
            drift_result   = None
            shap_result    = None
        plan = DriftAdvisor(MockReport()).advise()
        assert isinstance(plan, DriftPlan)
        assert plan.overall_status == "unknown"


# ── ShapAdvisor tests ──────────────────────────────────────────────────────

class TestShapAdvisor:

    def test_returns_shap_plan(self, full_report):
        plan = ShapAdvisor(full_report).advise()
        assert isinstance(plan, ShapPlan)

    def test_plan_has_model_name(self, full_report):
        plan = ShapAdvisor(full_report).advise()
        assert plan.model_name == "TestGBT"

    def test_top_features_not_empty(self, full_report):
        plan = ShapAdvisor(full_report).advise()
        assert len(plan.top_features) > 0

    def test_top_features_have_keys(self, full_report):
        plan = ShapAdvisor(full_report).advise()
        for f in plan.top_features:
            assert "name"          in f
            assert "mean_abs_shap" in f
            assert "rank"          in f

    def test_insights_are_list(self, full_report):
        plan = ShapAdvisor(full_report).advise()
        assert isinstance(plan.insights, list)

    def test_insight_priorities_valid(self, full_report):
        plan  = ShapAdvisor(full_report).advise()
        valid = {"high", "medium", "low"}
        for i in plan.insights:
            assert i.priority in valid

    def test_insight_categories_valid(self, full_report):
        plan  = ShapAdvisor(full_report).advise()
        valid = {"dominance", "redundancy", "engineering", "simplification"}
        for i in plan.insights:
            assert i.category in valid

    def test_summary_not_empty(self, full_report):
        plan = ShapAdvisor(full_report).advise()
        assert len(plan.summary) > 0

    def test_to_dict_keys(self, full_report):
        d = ShapAdvisor(full_report).advise().to_dict()
        assert "model_name"         in d
        assert "top_features"       in d
        assert "low_value_features" in d
        assert "summary"            in d

    def test_no_shap_result_returns_plan(self):
        class MockReport:
            model_name     = "Mock"
            fairness_result= None
            drift_result   = None
            shap_result    = None
        plan = ShapAdvisor(MockReport()).advise()
        assert isinstance(plan, ShapPlan)
        assert len(plan.insights) == 0
