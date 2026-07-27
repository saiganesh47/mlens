"""
tests/test_comparator.py
==========================
Unit tests for ModelComparator and ComparisonResult.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlens.comparison.model_comparator import ComparisonResult, ModelComparator


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    sensitive = np.random.choice(["A", "B"], size=len(y_test), replace=True)
    return X_train, X_test, y_train, y_test, sensitive


@pytest.fixture(scope="module")
def trained_models(data):
    X_train, _, _, y_train, _ = data
    # Fit all labels to X_train length
    y_tr = y_train[:len(X_train)]
    models = {
        "LogReg": LogisticRegression(max_iter=200, random_state=42).fit(X_train, y_tr),
        "RF":     RandomForestClassifier(n_estimators=20, random_state=42).fit(X_train, y_tr),
        "GBT":    GradientBoostingClassifier(n_estimators=20, random_state=42).fit(X_train, y_tr),
    }
    return models


@pytest.fixture(scope="module")
def comparison_result(trained_models, data):
    X_train, X_test, _, y_test, sensitive = data
    comparator = ModelComparator(
        models              = trained_models,
        X_train             = X_train,
        X_test              = X_test,
        y_test              = y_test,
        sensitive_features  = sensitive,
        shap_background_samples=50,
    )
    return comparator.compare()


# ── Tests ──────────────────────────────────────────────────────────────────

class TestModelComparator:

    def test_returns_comparison_result(self, comparison_result):
        assert isinstance(comparison_result, ComparisonResult)

    def test_all_models_in_result(self, comparison_result):
        assert set(comparison_result.model_names) == {"LogReg", "RF", "GBT"}

    def test_comparison_table_shape(self, comparison_result):
        assert len(comparison_result.comparison_table) == 3

    def test_all_reports_present(self, comparison_result):
        for name in ["LogReg", "RF", "GBT"]:
            assert name in comparison_result.audit_reports

    def test_runtime_positive(self, comparison_result):
        assert comparison_result.runtime_seconds > 0

    def test_timestamp_not_empty(self, comparison_result):
        assert len(comparison_result.timestamp) > 0


class TestComparisonResult:

    def test_best_model_returns_tuple(self, comparison_result):
        result = comparison_result.best_model(metric="dp_gap", higher_is_better=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_best_model_name_in_list(self, comparison_result):
        name, _ = comparison_result.best_model(metric="dp_gap", higher_is_better=False)
        assert name in ["LogReg", "RF", "GBT"]

    def test_best_model_invalid_metric_raises(self, comparison_result):
        with pytest.raises(ValueError):
            comparison_result.best_model(metric="nonexistent_metric")

    def test_rank_returns_dataframe(self, comparison_result):
        import pandas as pd
        ranked = comparison_result.rank(metric="dp_gap", higher_is_better=False)
        assert isinstance(ranked, pd.DataFrame)

    def test_rank_length_matches(self, comparison_result):
        ranked = comparison_result.rank(metric="dp_gap", higher_is_better=False)
        assert len(ranked) == 3

    def test_to_dict_keys(self, comparison_result):
        d = comparison_result.to_dict()
        assert "model_names"  in d
        assert "comparison"   in d
        assert "timestamp"    in d
        assert "runtime"      in d
