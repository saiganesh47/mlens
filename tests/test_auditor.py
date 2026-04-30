"""
tests/test_auditor.py
======================
Unit tests for the ModelAuditor orchestrator.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlens.auditor import AuditReport, ModelAuditor


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def trained_rf(classification_data):
    X_train, _, y_train, _ = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def auditor(trained_rf, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    sensitive = np.random.choice(["A", "B"], size=len(y_test), replace=True)
    return ModelAuditor(
        model=trained_rf,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        sensitive_features=sensitive,
        model_name="TestRF",
        shap_background_samples=50,
    )


# ── Tests ──────────────────────────────────────────────────────────────────

class TestModelAuditor:

    def test_run_returns_audit_report(self, auditor):
        report = auditor.run()
        assert isinstance(report, AuditReport)

    def test_report_has_model_name(self, auditor):
        report = auditor.run()
        assert report.model_name == "TestRF"

    def test_report_runtime_positive(self, auditor):
        report = auditor.run()
        assert report.runtime_seconds > 0

    def test_report_has_shap_result(self, auditor):
        report = auditor.run()
        assert report.shap_result is not None

    def test_report_has_fairness_result(self, auditor):
        report = auditor.run()
        assert report.fairness_result is not None

    def test_report_has_drift_result(self, auditor):
        report = auditor.run()
        assert report.drift_result is not None

    def test_summary_lines_not_empty(self, auditor):
        report = auditor.run()
        assert len(report.summary_lines) > 0

    def test_to_dict_serialisable(self, auditor):
        import json
        report = auditor.run()
        d = report.to_dict()
        assert json.dumps(d)  # must not raise

    def test_skip_shap(self, trained_rf, classification_data):
        X_train, X_test, _, y_test = classification_data
        auditor = ModelAuditor(
            model=trained_rf, X_train=X_train, X_test=X_test,
            y_test=y_test, run_shap=False,
        )
        report = auditor.run()
        assert report.shap_result is None

    def test_skip_drift(self, trained_rf, classification_data):
        X_train, X_test, _, y_test = classification_data
        auditor = ModelAuditor(
            model=trained_rf, X_train=X_train, X_test=X_test,
            y_test=y_test, run_drift=False,
        )
        report = auditor.run()
        assert report.drift_result is None

    def test_metadata_keys(self, auditor):
        report = auditor.run()
        assert "train_size"  in report.metadata
        assert "test_size"   in report.metadata
        assert "n_features"  in report.metadata
