"""
tests/test_drift.py
====================
Unit tests for DriftDetector and DriftResult.
"""

import numpy as np
import pytest

from mlens.drift.drift_detector import DriftDetector, DriftResult


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def stable_data():
    """Reference and production from the same distribution."""
    rng = np.random.default_rng(42)
    reference  = rng.normal(0, 1, (500, 5))
    production = rng.normal(0, 1, (200, 5))
    names = [f"feature_{i}" for i in range(5)]
    return reference, production, names


@pytest.fixture
def drifted_data():
    """Production has a large mean/variance shift on feature_0."""
    rng = np.random.default_rng(42)
    reference  = rng.normal(0, 1, (500, 5))
    production = rng.normal(0, 1, (200, 5))
    production[:, 0] = rng.normal(5, 3, 200)   # strong drift on feature_0
    names = [f"feature_{i}" for i in range(5)]
    return reference, production, names


@pytest.fixture
def constant_data():
    """Edge case: constant feature (no variance)."""
    reference  = np.ones((200, 3))
    production = np.ones((100, 3))
    return reference, production, ["c0", "c1", "c2"]


# ── Tests ──────────────────────────────────────────────────────────────────

class TestDriftDetector:

    def test_returns_drift_result(self, stable_data):
        ref, prod, names = stable_data
        result = DriftDetector(ref, names).detect(prod)
        assert isinstance(result, DriftResult)

    def test_stable_data_no_drift(self, stable_data):
        ref, prod, names = stable_data
        result = DriftDetector(ref, names).detect(prod)
        assert result.overall_status == "stable"
        assert result.n_drifted == 0

    def test_drifted_feature_detected(self, drifted_data):
        ref, prod, names = drifted_data
        result = DriftDetector(ref, names).detect(prod)
        drifted = result.drifted_features()
        assert "feature_0" in drifted

    def test_non_drifted_features_stable(self, drifted_data):
        ref, prod, names = drifted_data
        result = DriftDetector(ref, names).detect(prod)
        drifted = result.drifted_features()
        for i in range(1, 5):
            assert f"feature_{i}" not in drifted

    def test_feature_results_length(self, stable_data):
        ref, prod, names = stable_data
        result = DriftDetector(ref, names).detect(prod)
        assert len(result.feature_results) == 5

    def test_feature_result_keys(self, stable_data):
        ref, prod, names = stable_data
        result = DriftDetector(ref, names).detect(prod)
        required = {"feature", "psi", "psi_status", "ks_statistic", "ks_pvalue", "drifted"}
        for row in result.feature_results:
            assert required.issubset(set(row.keys()))

    def test_psi_non_negative(self, drifted_data):
        ref, prod, names = drifted_data
        result = DriftDetector(ref, names).detect(prod)
        for row in result.feature_results:
            assert row["psi"] >= 0

    def test_constant_feature_no_crash(self, constant_data):
        ref, prod, names = constant_data
        result = DriftDetector(ref, names).detect(prod)
        assert isinstance(result, DriftResult)

    def test_to_dict_keys(self, stable_data):
        ref, prod, names = stable_data
        d = DriftDetector(ref, names).detect(prod).to_dict()
        assert "overall_status"   in d
        assert "n_drifted"        in d
        assert "drifted_features" in d
        assert "details"          in d

    def test_psi_status_labels(self, stable_data):
        ref, prod, names = stable_data
        result = DriftDetector(ref, names).detect(prod)
        valid  = {"stable", "moderate", "significant"}
        for row in result.feature_results:
            assert row["psi_status"] in valid
