"""
mlens/automl/model_recommender.py
===================================
Analyses an AuditReport and recommends better model architectures
based on the findings — fairness gaps, drift severity, SHAP patterns,
and dataset characteristics.

Usage
-----
>>> from mlens.automl.model_recommender import ModelRecommender
>>> recommender = ModelRecommender(report, X_train, y_train)
>>> suggestions = recommender.recommend()
>>> for s in suggestions:
...     print(s.model_name, s.reason)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Recommendation dataclass ───────────────────────────────────────────────

@dataclass
class ModelSuggestion:
    """
    A single model recommendation.

    Attributes
    ----------
    model_name : str
        Human-readable model name.
    model_class : str
        Importable class string, e.g. 'sklearn.ensemble.RandomForestClassifier'.
    reason : str
        Plain-English explanation of why this model is suggested.
    priority : str
        'high' | 'medium' | 'low'
    hyperparams : dict
        Suggested starting hyperparameters.
    expected_improvement : str
        What metric is expected to improve.
    """
    model_name           : str
    model_class          : str
    reason               : str
    priority             : str
    hyperparams          : Dict[str, Any] = field(default_factory=dict)
    expected_improvement : str = ""


@dataclass
class RecommendationResult:
    """
    Full set of model recommendations from ModelRecommender.

    Attributes
    ----------
    current_model : str
        Name of the model that was audited.
    suggestions : list of ModelSuggestion
        Ranked list of recommended models (highest priority first).
    dataset_profile : dict
        Key characteristics of the training data.
    reasoning_summary : list of str
        Plain-English explanation of the recommendation logic.
    """
    current_model     : str
    suggestions       : List[ModelSuggestion]
    dataset_profile   : Dict[str, Any]
    reasoning_summary : List[str]

    def top(self, n: int = 3) -> List[ModelSuggestion]:
        """Return top-n suggestions by priority."""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            self.suggestions,
            key=lambda s: priority_order.get(s.priority, 3),
        )[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_model":      self.current_model,
            "dataset_profile":    self.dataset_profile,
            "reasoning_summary":  self.reasoning_summary,
            "suggestions": [
                {
                    "model":                s.model_name,
                    "class":                s.model_class,
                    "priority":             s.priority,
                    "reason":               s.reason,
                    "hyperparams":          s.hyperparams,
                    "expected_improvement": s.expected_improvement,
                }
                for s in self.suggestions
            ],
        }


# ── Recommender ────────────────────────────────────────────────────────────

class ModelRecommender:
    """
    Analyse an AuditReport and recommend better model architectures.

    The recommender looks at:
      - Dataset size and dimensionality
      - Fairness flags (if any → prefer fairness-aware models)
      - Drift severity (if high → prefer robust/adaptive models)
      - SHAP patterns (if few features dominate → prefer simpler models)
      - Current model type (to suggest something meaningfully different)

    Parameters
    ----------
    report : AuditReport
        Populated audit report from ModelAuditor.run().
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    """

    def __init__(
        self,
        report : Any,
        X_train: Any,
        y_train: Any,
    ) -> None:
        self.report  = report
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    # ---------------------------------------------------------------- public

    def recommend(self) -> RecommendationResult:
        """
        Analyse the report and return ranked model suggestions.

        Returns
        -------
        RecommendationResult
        """
        profile   = self._profile_dataset()
        reasoning : List[str] = []
        candidates: List[ModelSuggestion] = []

        # ── Fairness-driven recommendations ───────────────────────────────
        if self._has_fairness_issues():
            reasoning.append(
                "⚖️ Fairness flags detected — recommending fairness-aware "
                "and regularised models that reduce demographic disparity."
            )
            candidates += self._fairness_recommendations()

        # ── Drift-driven recommendations ───────────────────────────────────
        if self._has_drift_issues():
            reasoning.append(
                "📊 Significant feature drift detected — recommending "
                "robust models less sensitive to distribution shift."
            )
            candidates += self._drift_recommendations()

        # ── Dataset size recommendations ───────────────────────────────────
        n, d = self.X_train.shape
        if n > 50_000:
            reasoning.append(
                f"📦 Large dataset ({n:,} rows) — linear and boosting "
                "models scale better than tree ensembles at this size."
            )
            candidates += self._large_dataset_recommendations()
        elif n < 500:
            reasoning.append(
                f"📦 Small dataset ({n:,} rows) — prefer regularised "
                "models with strong priors to avoid overfitting."
            )
            candidates += self._small_dataset_recommendations()

        # ── SHAP sparsity recommendations ──────────────────────────────────
        if self._shap_is_sparse():
            reasoning.append(
                "🧠 SHAP analysis shows only a few dominant features — "
                "a simpler model may generalise better."
            )
            candidates += self._sparse_shap_recommendations()

        # ── Always include a strong baseline ──────────────────────────────
        candidates += self._baseline_recommendations()

        # Deduplicate by model_class, keep highest priority
        seen     : Dict[str, int] = {}
        deduped  : List[ModelSuggestion] = []
        p_rank   = {"high": 0, "medium": 1, "low": 2}
        for s in candidates:
            if s.model_class not in seen:
                seen[s.model_class] = p_rank.get(s.priority, 3)
                deduped.append(s)
            else:
                if p_rank.get(s.priority, 3) < seen[s.model_class]:
                    seen[s.model_class] = p_rank.get(s.priority, 3)
                    deduped = [x for x in deduped if x.model_class != s.model_class]
                    deduped.append(s)

        if not reasoning:
            reasoning.append(
                "✅ No critical issues detected. Suggestions are for "
                "potential performance improvements only."
            )

        return RecommendationResult(
            current_model     = self.report.model_name,
            suggestions       = sorted(deduped, key=lambda s: p_rank.get(s.priority, 3)),
            dataset_profile   = profile,
            reasoning_summary = reasoning,
        )

    # --------------------------------------------------------------- private

    def _has_fairness_issues(self) -> bool:
        fr = self.report.fairness_result
        return fr is not None and not fr.is_fair

    def _has_drift_issues(self) -> bool:
        dr = self.report.drift_result
        return dr is not None and dr.overall_status in ("moderate", "significant")

    def _shap_is_sparse(self) -> bool:
        """True if top-3 features account for >80% of total SHAP importance."""
        sr = self.report.shap_result
        if sr is None:
            return False
        vals  = sr.mean_abs_shap
        total = vals.sum()
        if total == 0:
            return False
        top3  = np.sort(vals)[::-1][:3].sum()
        return (top3 / total) > 0.80

    def _profile_dataset(self) -> Dict[str, Any]:
        n, d = self.X_train.shape
        classes, counts = np.unique(self.y_train, return_counts=True)
        imbalance_ratio = float(counts.min() / counts.max()) if len(counts) > 1 else 1.0
        return {
            "n_samples":       int(n),
            "n_features":      int(d),
            "n_classes":       int(len(classes)),
            "imbalance_ratio": round(imbalance_ratio, 3),
            "is_imbalanced":   imbalance_ratio < 0.3,
        }

    # ── Model catalogue ────────────────────────────────────────────────────

    def _fairness_recommendations(self) -> List[ModelSuggestion]:
        return [
            ModelSuggestion(
                model_name  = "Logistic Regression (L2)",
                model_class = "sklearn.linear_model.LogisticRegression",
                reason      = "Linear models with regularisation tend to have "
                              "smaller demographic parity gaps than complex trees.",
                priority    = "high",
                hyperparams = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"},
                expected_improvement = "Reduced demographic parity gap",
            ),
            ModelSuggestion(
                model_name  = "Fairlearn ExponentiatedGradient",
                model_class = "fairlearn.reductions.ExponentiatedGradient",
                reason      = "Fairness-constrained training directly minimises "
                              "demographic parity gap during optimisation.",
                priority    = "high",
                hyperparams = {"constraints": "DemographicParity"},
                expected_improvement = "Enforced fairness constraint",
            ),
            ModelSuggestion(
                model_name  = "Random Forest (balanced)",
                model_class = "sklearn.ensemble.RandomForestClassifier",
                reason      = "class_weight='balanced' improves fairness on "
                              "imbalanced group distributions.",
                priority    = "medium",
                hyperparams = {
                    "n_estimators": 200, "class_weight": "balanced",
                    "max_depth": 8, "random_state": 42,
                },
                expected_improvement = "Better recall for minority group",
            ),
        ]

    def _drift_recommendations(self) -> List[ModelSuggestion]:
        return [
            ModelSuggestion(
                model_name  = "XGBoost (with monotone constraints)",
                model_class = "xgboost.XGBClassifier",
                reason      = "Monotone constraints make XGBoost more robust "
                              "to feature distribution shifts.",
                priority    = "high",
                hyperparams = {
                    "n_estimators": 300, "learning_rate": 0.05,
                    "max_depth": 4, "subsample": 0.8,
                    "colsample_bytree": 0.8, "random_state": 42,
                },
                expected_improvement = "More stable under distribution shift",
            ),
            ModelSuggestion(
                model_name  = "LightGBM (dart boosting)",
                model_class = "lightgbm.LGBMClassifier",
                reason      = "DART boosting drops trees randomly during training, "
                              "producing models more robust to concept drift.",
                priority    = "medium",
                hyperparams = {
                    "boosting_type": "dart", "n_estimators": 200,
                    "learning_rate": 0.05, "random_state": 42,
                },
                expected_improvement = "Reduced sensitivity to drifted features",
            ),
        ]

    def _large_dataset_recommendations(self) -> List[ModelSuggestion]:
        return [
            ModelSuggestion(
                model_name  = "SGD Classifier",
                model_class = "sklearn.linear_model.SGDClassifier",
                reason      = "Online learning with SGD scales to millions of "
                              "rows with constant memory usage.",
                priority    = "medium",
                hyperparams = {"loss": "log_loss", "max_iter": 100, "random_state": 42},
                expected_improvement = "Faster training at scale",
            ),
            ModelSuggestion(
                model_name  = "HistGradientBoosting",
                model_class = "sklearn.ensemble.HistGradientBoostingClassifier",
                reason      = "Histogram-based boosting is 10-100x faster than "
                              "standard GBT on large datasets.",
                priority    = "high",
                hyperparams = {"max_iter": 200, "learning_rate": 0.05, "random_state": 42},
                expected_improvement = "10x faster training",
            ),
        ]

    def _small_dataset_recommendations(self) -> List[ModelSuggestion]:
        return [
            ModelSuggestion(
                model_name  = "SVM (RBF kernel)",
                model_class = "sklearn.svm.SVC",
                reason      = "SVMs generalise well on small datasets with "
                              "high-dimensional features.",
                priority    = "medium",
                hyperparams = {"C": 1.0, "kernel": "rbf", "probability": True},
                expected_improvement = "Better generalisation on small data",
            ),
            ModelSuggestion(
                model_name  = "Extra Trees",
                model_class = "sklearn.ensemble.ExtraTreesClassifier",
                reason      = "Extra randomness in splits reduces overfitting "
                              "on small datasets.",
                priority    = "medium",
                hyperparams = {"n_estimators": 200, "random_state": 42},
                expected_improvement = "Reduced overfitting",
            ),
        ]

    def _sparse_shap_recommendations(self) -> List[ModelSuggestion]:
        return [
            ModelSuggestion(
                model_name  = "Decision Tree (shallow)",
                model_class = "sklearn.tree.DecisionTreeClassifier",
                reason      = "When 3 features drive 80%+ of predictions, a "
                              "shallow tree is interpretable and competitive.",
                priority    = "low",
                hyperparams = {"max_depth": 5, "random_state": 42},
                expected_improvement = "Maximum interpretability",
            ),
        ]

    def _baseline_recommendations(self) -> List[ModelSuggestion]:
        return [
            ModelSuggestion(
                model_name  = "CatBoost",
                model_class = "catboost.CatBoostClassifier",
                reason      = "CatBoost handles categorical features natively "
                              "and often outperforms XGBoost with less tuning.",
                priority    = "medium",
                hyperparams = {"iterations": 300, "learning_rate": 0.05,
                               "depth": 6, "verbose": 0},
                expected_improvement = "Potential accuracy improvement",
            ),
        ]
