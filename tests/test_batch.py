"""
tests/test_batch.py
=====================
Unit tests for BatchAuditor and BatchAuditSummary.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlens.batch.batch_auditor import BatchAuditor, BatchAuditSummary


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def models(data):
    X_train, _, y_train, _ = data
    return {
        "LR":  LogisticRegression(max_iter=200).fit(X_train, y_train),
        "RF":  RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train),
    }


@pytest.fixture(scope="module")
def batch_summary(models, data, tmp_path_factory):
    X_train, X_test, _, y_test = data
    out_dir = str(tmp_path_factory.mktemp("reports"))
    batch   = BatchAuditor(
        models      = models,
        X_test      = X_test,
        y_test      = y_test,
        X_train     = X_train,
        output_dir  = out_dir,
        max_workers = 2,
        run_shap    = True,
        run_fairness= False,
        run_drift   = True,
        shap_background_samples=30,
    )
    return batch.run()


# ── Tests ──────────────────────────────────────────────────────────────────

class TestBatchAuditor:

    def test_raises_without_source(self, data):
        _, X_test, _, y_test = data
        with pytest.raises(ValueError):
            BatchAuditor(X_test=X_test, y_test=y_test)

    def test_returns_batch_summary(self, batch_summary):
        assert isinstance(batch_summary, BatchAuditSummary)

    def test_n_models_correct(self, batch_summary):
        assert batch_summary.n_models == 2

    def test_n_success_correct(self, batch_summary):
        assert batch_summary.n_success == 2

    def test_n_failed_zero(self, batch_summary):
        assert batch_summary.n_failed == 0

    def test_summary_table_rows(self, batch_summary):
        assert len(batch_summary.summary_table) == 2

    def test_runtime_positive(self, batch_summary):
        assert batch_summary.runtime_seconds > 0

    def test_output_dir_set(self, batch_summary):
        assert len(batch_summary.output_dir) > 0

    def test_to_dict_keys(self, batch_summary):
        d = batch_summary.to_dict()
        assert "n_models"    in d
        assert "n_success"   in d
        assert "n_failed"    in d
        assert "summary"     in d

    def test_save_csv(self, batch_summary, tmp_path):
        path = str(tmp_path / "out.csv")
        batch_summary.save_csv(path)
        import os
        assert os.path.exists(path)

    def test_drift_in_table(self, batch_summary):
        assert "drift_status" in batch_summary.summary_table.columns
