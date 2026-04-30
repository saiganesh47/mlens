"""
tests/test_fairness.py
=======================
Unit tests for FairnessEvaluator and FairnessResult.
"""

import numpy as np
import pytest

from mlens.fairness.fairness_metrics import FairnessEvaluator, FairnessResult


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def biased_data():
    """Synthetic data with clear demographic parity violation."""
    np.random.seed(42)
    n = 400
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    sensitive = np.array(["A"] * 200 + ["B"] * 200)
    # Introduce bias: flip 40% of group B predictions
    flip_idx = np.where(sensitive == "B")[0][:80]
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    return y_true, y_pred, sensitive


@pytest.fixture
def fair_data():
    """Synthetic data with no demographic parity violation."""
    np.random.seed(0)
    n = 400
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    sensitive = np.array(["A"] * 200 + ["B"] * 200)
    return y_true, y_pred, sensitive


# ── Tests ──────────────────────────────────────────────────────────────────

class TestFairnessEvaluator:

    def test_returns_fairness_result(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        assert isinstance(result, FairnessResult)

    def test_biased_data_flags(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        assert len(result.flags) > 0

    def test_fair_data_no_flags(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        assert result.is_fair

    def test_dp_gap_range(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        assert 0.0 <= result.demographic_parity_gap <= 1.0

    def test_disparate_impact_range(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        assert 0.0 <= result.disparate_impact <= 1.0

    def test_per_group_metrics_not_empty(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        assert len(result.per_group_metrics) == 2  # groups A and B

    def test_per_group_has_required_keys(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        result = ev.evaluate()
        required = {"group", "accuracy", "precision", "recall", "f1"}
        for row in result.per_group_metrics:
            assert required.issubset(set(row.keys()))

    def test_to_dict_keys(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        ev = FairnessEvaluator(y_true, y_pred, sensitive)
        d  = ev.evaluate().to_dict()
        assert "demographic_parity_gap" in d
        assert "equalized_odds_gap"     in d
        assert "disparate_impact"       in d
        assert "flags"                  in d

    def test_custom_thresholds(self, biased_data):
        """Stricter thresholds should generate more flags."""
        y_true, y_pred, sensitive = biased_data
        strict = FairnessEvaluator(y_true, y_pred, sensitive,
                                   dp_threshold=0.01,
                                   eo_threshold=0.01,
                                   di_threshold=0.99)
        lenient = FairnessEvaluator(y_true, y_pred, sensitive,
                                    dp_threshold=0.99,
                                    eo_threshold=0.99,
                                    di_threshold=0.01)
        assert len(strict.evaluate().flags) >= len(lenient.evaluate().flags)
