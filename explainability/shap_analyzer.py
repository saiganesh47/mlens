"""
mlens/explainability/shap_analyzer.py
======================================
Wraps the `shap` library to produce global and local explanations
for any sklearn-compatible or tree-based model.

Supports:
  - TreeExplainer  → XGBoost, LightGBM, RandomForest, ExtraTrees
  - LinearExplainer → LogisticRegression, LinearSVC, Ridge
  - KernelExplainer → fallback for any black-box model (slower)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import shap


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ShapResult:
    """
    Holds raw SHAP values and derived summaries.

    Attributes
    ----------
    shap_values : np.ndarray of shape (n_samples, n_features)
        SHAP values for every test observation.
    base_value : float
        Expected model output (SHAP baseline).
    feature_names : list of str
        Ordered feature names matching columns in shap_values.
    mean_abs_shap : np.ndarray of shape (n_features,)
        Mean |SHAP| per feature — used for global importance ranking.
    """

    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    mean_abs_shap: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

    def top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return the top-n most important features sorted by mean |SHAP|.

        Returns
        -------
        list of dicts with keys: 'rank', 'name', 'mean_abs_shap'
        """
        indices = np.argsort(self.mean_abs_shap)[::-1][:n]
        return [
            {
                "rank": i + 1,
                "name": self.feature_names[idx] if self.feature_names else f"f{idx}",
                "mean_abs_shap": round(float(self.mean_abs_shap[idx]), 6),
            }
            for i, idx in enumerate(indices)
        ]

    def local_explanation(self, sample_idx: int) -> List[Dict[str, Any]]:
        """
        Return per-feature SHAP contributions for a single observation.

        Parameters
        ----------
        sample_idx : int
            Row index into shap_values to explain.
        """
        row = self.shap_values[sample_idx]
        pairs = zip(
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(row))],
            row,
        )
        return [
            {"feature": name, "shap_value": round(float(val), 6)}
            for name, val in sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        ]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ShapAnalyzer:
    """
    Selects and runs the most efficient SHAP explainer for the given model.

    Parameters
    ----------
    model : Any
        Trained model to explain.
    background_data : np.ndarray
        A representative sample of training data used as the SHAP background.
        Keep ≤ 200 rows for performance; for KernelExplainer ≤ 50 rows.
    feature_names : list of str, optional
        Human-readable feature names.
    """

    # Model types that benefit from TreeExplainer
    _TREE_TYPES = (
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
    )

    # Model types that benefit from LinearExplainer
    _LINEAR_TYPES = (
        "LogisticRegression", "LinearSVC",
        "Ridge", "Lasso", "ElasticNet", "SGDClassifier",
    )

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self._explainer: Optional[Any] = None

    # ---------------------------------------------------------------- public

    def explain(self, X_test: np.ndarray) -> ShapResult:
        """
        Compute SHAP values for all rows in X_test.

        Parameters
        ----------
        X_test : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        ShapResult
        """
        explainer = self._build_explainer()
        raw = explainer.shap_values(X_test)

        # Binary classifiers return a list [neg_class_vals, pos_class_vals]
        shap_vals = self._extract_positive_class(raw)

        base = self._extract_base_value(explainer)

        return ShapResult(
            shap_values=shap_vals,
            base_value=float(base),
            feature_names=self.feature_names or [f"feature_{i}" for i in range(shap_vals.shape[1])],
        )

    # --------------------------------------------------------------- private

    def _build_explainer(self) -> Any:
        """Instantiate the best SHAP explainer for self.model."""
        if self._explainer is not None:
            return self._explainer

        model_type = type(self.model).__name__

        if model_type in self._TREE_TYPES:
            self._explainer = shap.TreeExplainer(
                self.model,
                data=self.background_data,
                feature_perturbation="interventional",
            )
        elif model_type in self._LINEAR_TYPES:
            self._explainer = shap.LinearExplainer(
                self.model,
                masker=shap.maskers.Independent(self.background_data),
            )
        else:
            # Generic fallback — works on any model with .predict_proba / .predict
            predict_fn = self._get_predict_fn()
            # Use a small kmeans summary to keep KernelExplainer tractable
            background_summary = shap.kmeans(self.background_data, min(50, len(self.background_data)))
            self._explainer = shap.KernelExplainer(predict_fn, background_summary)

        return self._explainer

    def _get_predict_fn(self):
        """Return a callable suitable for KernelExplainer."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        return self.model.predict

    @staticmethod
    def _extract_positive_class(raw: Any) -> np.ndarray:
        """Handle both single-output and multi-output SHAP returns."""
        if isinstance(raw, list) and len(raw) == 2:
            return np.array(raw[1])
        if isinstance(raw, list):
            return np.array(raw[0])
        return np.array(raw)

    @staticmethod
    def _extract_base_value(explainer: Any) -> float:
        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            return float(bv[-1])
        return float(bv)
