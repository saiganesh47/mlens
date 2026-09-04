"""
mlens/recommendations/shap_advisor.py
=======================================
Interprets SHAP results and generates feature engineering
suggestions, redundancy warnings, and model simplification advice.

Usage
-----
>>> from mlens.recommendations.shap_advisor import ShapAdvisor
>>> advisor = ShapAdvisor(report)
>>> plan = advisor.advise()
>>> for insight in plan.insights:
...     print(insight.title, insight.explanation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Insight dataclass ──────────────────────────────────────────────────────

@dataclass
class ShapInsight:
    """
    A single SHAP-derived insight or recommendation.
    """
    category    : str   # 'dominance' | 'redundancy' | 'engineering' | 'simplification'
    priority    : str   # 'high' | 'medium' | 'low'
    title       : str
    explanation : str
    suggestion  : str
    code_snippet: str   = ""


@dataclass
class ShapPlan:
    """
    Full SHAP advisory plan.

    Attributes
    ----------
    model_name : str
    top_features : list of dict   Top features by mean |SHAP|.
    low_value_features : list     Features contributing <1% of total SHAP.
    insights : list of ShapInsight
    summary : str
    """
    model_name         : str
    top_features       : List[Dict[str, Any]]
    low_value_features : List[str]
    insights           : List[ShapInsight]
    summary            : str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":         self.model_name,
            "top_features":       self.top_features,
            "low_value_features": self.low_value_features,
            "n_insights":         len(self.insights),
            "summary":            self.summary,
        }


# ── Advisor ────────────────────────────────────────────────────────────────

class ShapAdvisor:
    """
    Generate feature engineering recommendations from SHAP results.

    Parameters
    ----------
    report : AuditReport
        Populated report from ModelAuditor.run().
    dominance_threshold : float
        Top-K features SHAP fraction above which to flag dominance (default: 0.80).
    low_value_threshold : float
        Feature SHAP fraction below which to flag as low-value (default: 0.01).
    """

    def __init__(
        self,
        report               : Any,
        dominance_threshold  : float = 0.80,
        low_value_threshold  : float = 0.01,
    ) -> None:
        self.report              = report
        self.dominance_threshold = dominance_threshold
        self.low_value_threshold = low_value_threshold

    # ---------------------------------------------------------------- public

    def advise(self) -> ShapPlan:
        """
        Analyse SHAP results and return a ShapPlan.

        Returns
        -------
        ShapPlan
        """
        sr = self.report.shap_result
        if sr is None:
            return ShapPlan(
                model_name         = self.report.model_name,
                top_features       = [],
                low_value_features = [],
                insights           = [],
                summary            = "SHAP analysis was not run. "
                                     "Enable with run_shap=True.",
            )

        top         = sr.top_features(n=20)
        total_shap  = float(sr.mean_abs_shap.sum())
        insights    : List[ShapInsight] = []

        # ── Feature dominance ──────────────────────────────────────────────
        cumsum = 0.0
        k_feats: List[str] = []
        for f in top:
            cumsum += f["mean_abs_shap"]
            k_feats.append(f["name"])
            if total_shap > 0 and cumsum / total_shap >= self.dominance_threshold:
                break

        if len(k_feats) <= 3 and len(top) > 5:
            insights.append(self._dominance_insight(k_feats, cumsum, total_shap))

        # ── Low-value features ─────────────────────────────────────────────
        low_value: List[str] = [
            f["name"] for f in top
            if total_shap > 0
            and (f["mean_abs_shap"] / total_shap) < self.low_value_threshold
        ]
        if low_value:
            insights.append(self._low_value_insight(low_value))

        # ── Feature interaction suggestion ────────────────────────────────
        if len(top) >= 2:
            insights.append(self._interaction_insight(top[:2]))

        # ── Simplification suggestion ──────────────────────────────────────
        if len(k_feats) <= 5:
            insights.append(self._simplification_insight(k_feats))

        # ── Negative SHAP features ─────────────────────────────────────────
        neg_features = self._find_negative_contributors(sr)
        if neg_features:
            insights.append(self._negative_contributor_insight(neg_features))

        # Sort by priority
        order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda i: order.get(i.priority, 3))

        summary = self._build_summary(top, low_value, k_feats, total_shap)

        return ShapPlan(
            model_name         = self.report.model_name,
            top_features       = top[:10],
            low_value_features = low_value,
            insights           = insights,
            summary            = summary,
        )

    # ── Insight builders ───────────────────────────────────────────────────

    def _dominance_insight(
        self, k_feats: List[str], cumsum: float, total: float
    ) -> ShapInsight:
        pct = int(100 * cumsum / total) if total > 0 else 0
        return ShapInsight(
            category    = "dominance",
            priority    = "high",
            title       = f"Feature dominance: {len(k_feats)} features drive {pct}% of predictions",
            explanation = (
                f"Features {k_feats} account for {pct}% of total SHAP importance. "
                "This level of dominance risks overfitting to these features "
                "and makes the model brittle if they drift."
            ),
            suggestion  = (
                "Consider (a) engineering interaction terms between these features, "
                "(b) collecting additional predictive features, or "
                "(c) using a simpler model that explicitly relies on these few features."
            ),
            code_snippet = f"""# Investigate dominant features more deeply
dominant = {k_feats}

# Option A: Create polynomial interactions
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_train[dominant])

# Option B: Partial dependence plot for the top feature
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_test, [0])
""",
        )

    def _low_value_insight(self, low_value: List[str]) -> ShapInsight:
        feat_str = '", "'.join(low_value[:5])
        return ShapInsight(
            category    = "redundancy",
            priority    = "medium",
            title       = f"Remove {len(low_value)} low-value features",
            explanation = (
                f"{len(low_value)} features contribute <1% of total SHAP importance. "
                "Removing them reduces model complexity and training time "
                "with minimal accuracy impact."
            ),
            suggestion  = (
                "Drop these features and retrain. Check if any are proxies "
                "for sensitive attributes before removing."
            ),
            code_snippet = f"""# Features contributing < 1% of SHAP importance
low_value_features = ["{feat_str}"]

X_train_slim = X_train.drop(columns=low_value_features, errors="ignore")
X_test_slim  = X_test.drop(columns=low_value_features,  errors="ignore")

model.fit(X_train_slim, y_train)
score = model.score(X_test_slim, y_test)
print(f"Score after removing low-value features: {{score:.4f}}")
""",
        )

    def _interaction_insight(self, top2: List[Dict]) -> ShapInsight:
        f1, f2 = top2[0]["name"], top2[1]["name"]
        return ShapInsight(
            category    = "engineering",
            priority    = "medium",
            title       = f"Create interaction feature: {f1} × {f2}",
            explanation = (
                f"The top two SHAP features are '{f1}' and '{f2}'. "
                "An explicit interaction term can help the model capture "
                "their joint effect more efficiently."
            ),
            suggestion  = "Create a ratio, product, or bin-crossed feature.",
            code_snippet = f"""import pandas as pd

# Product interaction
df["{f1}_x_{f2}"] = df["{f1}"] * df["{f2}"]

# Or ratio (guard against division by zero)
df["{f1}_div_{f2}"] = df["{f1}"] / (df["{f2}"] + 1e-8)

# Retrain with new feature
X_train_new = X_train.copy()
X_train_new["{f1}_x_{f2}"] = X_train["{f1}"] * X_train["{f2}"]
model.fit(X_train_new, y_train)
""",
        )

    def _simplification_insight(self, k_feats: List[str]) -> ShapInsight:
        return ShapInsight(
            category    = "simplification",
            priority    = "low",
            title       = f"Consider a simpler model using only {len(k_feats)} features",
            explanation = (
                f"Since {k_feats} dominate predictions, a Logistic Regression "
                "or shallow Decision Tree on these features alone may achieve "
                "similar accuracy with much better interpretability."
            ),
            suggestion  = "Try a feature-selected simple model as a baseline.",
            code_snippet = f"""from sklearn.linear_model import LogisticRegression

top_features = {k_feats}
X_simple = X_train[top_features]
X_test_simple = X_test[top_features]

simple_model = LogisticRegression(max_iter=1000, random_state=42)
simple_model.fit(X_simple, y_train)
score = simple_model.score(X_test_simple, y_test)
print(f"Simple model score ({{len(top_features)}} features): {{score:.4f}}")
""",
        )

    def _negative_contributor_insight(self, neg_features: List[str]) -> ShapInsight:
        feat_str = ", ".join(neg_features[:3])
        return ShapInsight(
            category    = "engineering",
            priority    = "low",
            title       = f"Investigate negative contributors: {feat_str}",
            explanation = (
                "Some features consistently push predictions in the wrong direction. "
                "These may be noise features or proxies for the outcome "
                "that confuse the model."
            ),
            suggestion  = (
                "Examine these features closely — consider transforming, "
                "binning, or removing them."
            ),
            code_snippet = f"""# Check correlation of negative contributors with target
import pandas as pd

neg_features = {neg_features[:3]}
corr = pd.DataFrame(X_train, columns=feature_names)[neg_features].corrwith(
    pd.Series(y_train)
)
print(corr)
""",
        )

    def _find_negative_contributors(self, sr: Any) -> List[str]:
        """Find features whose mean SHAP value (not abs) is consistently negative."""
        if not hasattr(sr, "shap_values"):
            return []
        mean_shap    = sr.shap_values.mean(axis=0)
        names        = sr.feature_names or [f"f{i}" for i in range(len(mean_shap))]
        neg_features = [
            names[i] for i, v in enumerate(mean_shap) if v < -0.01
        ]
        return neg_features[:5]

    @staticmethod
    def _build_summary(
        top: List[Dict], low_value: List[str],
        k_feats: List[str], total: float,
    ) -> str:
        top3_names = [f["name"] for f in top[:3]]
        return (
            f"Top 3 features: {', '.join(top3_names)}. "
            f"{len(low_value)} low-value feature(s) identified for removal. "
            f"{len(k_feats)} feature(s) drive the majority of predictions."
        )
