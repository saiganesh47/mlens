"""
tests/test_api.py
==================
Integration tests for the MLens REST API using FastAPI TestClient.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_payload():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (200, 5)).tolist()
    X_test  = rng.normal(0, 1, (50,  5)).tolist()
    y_test  = rng.integers(0, 2, 50).tolist()
    sensitive = (["A"] * 25 + ["B"] * 25)
    return {
        "X_train":           X_train,
        "X_test":            X_test,
        "y_test":            [float(v) for v in y_test],
        "sensitive_features": sensitive,
        "feature_names":     [f"f{i}" for i in range(5)],
        "model_name":        "TestModel",
        "run_shap":          True,
        "run_fairness":      True,
        "run_drift":         True,
        "shap_background_samples": 50,
    }


@pytest.fixture
def fairness_payload():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n).tolist()
    y_pred = [int(v) for v in y_true]
    sensitive = ["A"] * 100 + ["B"] * 100
    return {
        "y_true":             [float(v) for v in y_true],
        "y_pred":             [float(v) for v in y_pred],
        "sensitive_features": sensitive,
        "sensitive_feature_name": "group",
    }


@pytest.fixture
def drift_payload():
    rng = np.random.default_rng(42)
    return {
        "X_reference":  rng.normal(0, 1, (300, 4)).tolist(),
        "X_production": rng.normal(0, 1, (100, 4)).tolist(),
        "feature_names": ["a", "b", "c", "d"],
    }


# ── Health tests ───────────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_endpoints(self):
        r = client.get("/")
        assert "endpoints" in r.json()

    def test_health_returns_healthy(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health_has_version(self):
        r = client.get("/health")
        assert "version" in r.json()

    def test_health_has_timestamp(self):
        r = client.get("/health")
        assert "timestamp" in r.json()

    def test_version_endpoint(self):
        r = client.get("/version")
        assert r.status_code == 200
        assert r.json()["version"] == "0.3.0"


# ── Full audit tests ───────────────────────────────────────────────────────

class TestFullAuditEndpoint:

    def test_returns_200(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert r.status_code == 200

    def test_response_has_model_name(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert r.json()["model_name"] == "TestModel"

    def test_response_has_summary(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert len(r.json()["summary"]) > 0

    def test_response_has_shap(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert r.json()["shap"] is not None

    def test_response_has_fairness(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert r.json()["fairness"] is not None

    def test_response_has_drift(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert r.json()["drift"] is not None

    def test_skip_shap(self, sample_payload):
        payload = {**sample_payload, "run_shap": False}
        r = client.post("/audit/", json=payload)
        assert r.json()["shap"] is None

    def test_skip_drift(self, sample_payload):
        payload = {**sample_payload, "run_drift": False}
        r = client.post("/audit/", json=payload)
        assert r.json()["drift"] is None

    def test_mismatched_features_returns_422(self, sample_payload):
        payload = {**sample_payload,
                   "X_test": [[1.0, 2.0]]}  # 2 features vs 5
        r = client.post("/audit/", json=payload)
        assert r.status_code == 422

    def test_runtime_seconds_positive(self, sample_payload):
        r = client.post("/audit/", json=sample_payload)
        assert r.json()["runtime_seconds"] > 0


# ── Fairness-only tests ────────────────────────────────────────────────────

class TestFairnessEndpoint:

    def test_returns_200(self, fairness_payload):
        r = client.post("/audit/fairness", json=fairness_payload)
        assert r.status_code == 200

    def test_has_dp_gap(self, fairness_payload):
        r = client.post("/audit/fairness", json=fairness_payload)
        assert "demographic_parity_gap" in r.json()

    def test_has_per_group_metrics(self, fairness_payload):
        r = client.post("/audit/fairness", json=fairness_payload)
        assert len(r.json()["per_group_metrics"]) == 2


# ── Drift-only tests ───────────────────────────────────────────────────────

class TestDriftEndpoint:

    def test_returns_200(self, drift_payload):
        r = client.post("/audit/drift", json=drift_payload)
        assert r.status_code == 200

    def test_has_overall_status(self, drift_payload):
        r = client.post("/audit/drift", json=drift_payload)
        assert r.json()["overall_status"] in ("stable", "moderate", "significant")

    def test_details_length_matches_features(self, drift_payload):
        r = client.post("/audit/drift", json=drift_payload)
        assert len(r.json()["details"]) == 4
