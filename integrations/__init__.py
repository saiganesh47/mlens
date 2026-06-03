"""
tests/test_pytorch.py
======================
Unit tests for PyTorchExplainer.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.nn as nn

from mlens.explainability.pytorch_explainer import PyTorchExplainer, PyTorchShapResult


# ── Simple test model ──────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """Tiny 2-layer MLP for testing."""
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mlp_model():
    model = SimpleMLP(input_dim=10)
    model.eval()
    return model


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (100, 10)).astype(np.float32)
    X_test  = rng.normal(0, 1, (20,  10)).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(10)]
    return X_train, X_test, feature_names


# ── Tests ──────────────────────────────────────────────────────────────────

class TestPyTorchExplainer:

    def test_returns_pytorch_shap_result(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        assert isinstance(result, PyTorchShapResult)

    def test_shap_values_shape(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        assert result.shap_values.shape == (len(X_test), X_test.shape[1])

    def test_mean_abs_shap_non_negative(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        assert (result.mean_abs_shap >= 0).all()

    def test_top_features_length(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        assert len(result.top_features(n=5)) == 5

    def test_top_features_sorted(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        top = result.top_features(n=10)
        values = [f["mean_abs_shap"] for f in top]
        assert values == sorted(values, reverse=True)

    def test_feature_names_in_result(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        assert result.feature_names == names

    def test_local_explanation(self, mlp_model, data):
        X_train, X_test, names = data
        explainer = PyTorchExplainer(mlp_model, X_train[:50], feature_names=names)
        result = explainer.explain(X_test)
        local = result.local_explanation(0)
        assert len(local) == X_test.shape[1]
        assert "feature" in local[0]
        assert "shap_value" in local[0]
