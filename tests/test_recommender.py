"""
tests/test_recommender.py
===========================
Unit tests for ModelRecommender and RecommendationResult.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlens.automl.model_recommender import (
    ModelRecommender, ModelSuggestion, RecommendationResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def report_with_issues(data):
    """Report with fairness + drift issues."""
    X_train, X_test, y_train, y_test = data
    model = GradientBoostingClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    sensitive = np.random.choice(["A", "B"], size=len(y_test), replace=True)

    from mlens.auditor import ModelAuditor
    return ModelAuditor(
        model=model, X_train=X_train, X_test=X_test,
        y_test=y_test, sensitive_features=sensitive,
        model_name="GBT-Test", shap_background_samples=30,
    ).run()


@pytest.fixture(scope="module")
def recommendation(report_with_issues, data):
    X_train, _, y_train, _ = data
    return ModelRecommender(
        report=report_with_issues,
        X_train=X_train,
        y_train=y_train,
    ).recommend()


# ── Tests ──────────────────────────────────────────────────────────────────

class TestModelRecommender:

    def test_returns_recommendation_result(self, recommendation):
        assert isinstance(recommendation, RecommendationResult)

    def test_suggestions_not_empty(self, recommendation):
        assert len(recommendation.suggestions) > 0

    def test_all_suggestions_are_model_suggestion(self, recommendation):
        for s in recommendation.suggestions:
            assert isinstance(s, ModelSuggestion)

    def test_suggestions_have_priority(self, recommendation):
        valid = {"high", "medium", "low", "critical"}
        for s in recommendation.suggestions:
            assert s.priority in valid

    def test_suggestions_have_model_class(self, recommendation):
        for s in recommendation.suggestions:
            assert len(s.model_class) > 0

    def test_suggestions_have_reason(self, recommendation):
        for s in recommendation.suggestions:
            assert len(s.reason) > 0

    def test_reasoning_summary_not_empty(self, recommendation):
        assert len(recommendation.reasoning_summary) > 0

    def test_dataset_profile_keys(self, recommendation):
        profile = recommendation.dataset_profile
        assert "n_samples"       in profile
        assert "n_features"      in profile
        assert "n_classes"       in profile
        assert "imbalance_ratio" in profile

    def test_top_returns_correct_n(self, recommendation):
        top3 = recommendation.top(n=3)
        assert len(top3) <= 3

    def test_top_sorted_by_priority(self, recommendation):
        order = {"high": 0, "medium": 1, "low": 2, "critical": -1}
        top   = recommendation.top(n=10)
        ranks = [order.get(s.priority, 3) for s in top]
        assert ranks == sorted(ranks)

    def test_no_duplicate_model_classes(self, recommendation):
        classes = [s.model_class for s in recommendation.suggestions]
        assert len(classes) == len(set(classes))

    def test_to_dict_keys(self, recommendation):
        d = recommendation.to_dict()
        assert "current_model"     in d
        assert "suggestions"       in d
        assert "dataset_profile"   in d
        assert "reasoning_summary" in d
