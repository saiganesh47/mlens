"""
mlens/fairness/fairness_metrics.py
====================================
Computes group-level fairness metrics using the `fairlearn` library.

Metrics computed
----------------
- Demographic Parity Difference  (selection rate gap across groups)
- Equalized Odds Difference       (TPR + FPR gap across groups)
- Per-group accuracy, precision, recall, F1
- Disparate Impact ratio          (EEOC 4/5ths rule)

References
----------
- Fairlearn: https://fairlearn.org/
- EEOC Uniform Guidelines on Employee Selection Procedures (1978)
- Hardt et al., "Equality of Opportunity in Supervised Learning" (NeurIPS 2016)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FairnessResult:
    """
    Fairness diagnostics for one sensitive attribute.

    Attributes
    ----------
    sensitive_feature_name : str
        Name of the protected attribute evaluated.
    demographic_parity_gap : float
        Max difference in selection rate across groups.
        Values > 0.10 typically warrant investigation.
    equalized_odds_gap : float
        Max difference in (TPR, FPR) across groups.
    disparate_impact : float
        Ratio of lowest to highest selection rate (EEOC 4/5ths rule).
        Values < 0.80 may indicate adverse impact.
    per_group_metrics : list of dict
        Per-group breakdown: accuracy, precision, recall, F1,
        selection_rate, FPR, FNR.
    flags : list of str
        Human-readable warning strings for metrics that exceed thresholds.
    """

    sensitive_feature_name: str
    demographic_parity_gap: float
    equalized_odds_gap: float
    disparate_impact: float
    per_group_metrics: List[Dict[str, Any]]
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensitive_feature": self.sensitive_feature_name,
            "demographic_parity_gap": round(self.demographic_parity_gap, 4),
            "equalized_odds_gap": round(self.equalized_odds_gap, 4),
            "disparate_impact": round(self.disparate_impact, 4),
            "flags": self.flags,
            "per_group_metrics": self.per_group_metrics,
        }

    @property
    def is_fair(self) -> bool:
        """Rough heuristic: True if no threshold violations detected."""
        return len(self.flags) == 0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FairnessEvaluator:
    """
    Compute fairness metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Model predictions (binary).
    sensitive_features : pd.Series | array-like
        Protected attribute values aligned with y_true / y_pred.
    sensitive_feature_name : str
        Display name for the attribute (default: 'sensitive_feature').
    dp_threshold : float
        Demographic parity gap threshold for flagging (default: 0.10).
    eo_threshold : float
        Equalized odds gap threshold for flagging (default: 0.10).
    di_threshold : float
        Disparate impact ratio below which to flag (default: 0.80).
    """

    def __init__(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
        sensitive_feature_name: str = "sensitive_feature",
        dp_threshold: float = 0.10,
        eo_threshold: float = 0.10,
        di_threshold: float = 0.80,
    ) -> None:
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.sensitive_features = np.asarray(sensitive_features)
        self.sensitive_feature_name = sensitive_feature_name
        self.dp_threshold = dp_threshold
        self.eo_threshold = eo_threshold
        self.di_threshold = di_threshold

    # ---------------------------------------------------------------- public

    def evaluate(self) -> FairnessResult:
        """
        Run all fairness checks and return a FairnessResult.
        """
        dp_gap = demographic_parity_difference(
            self.y_true, self.y_pred,
            sensitive_features=self.sensitive_features,
        )
        eo_gap = equalized_odds_difference(
            self.y_true, self.y_pred,
            sensitive_features=self.sensitive_features,
        )
        di = self._disparate_impact()
        per_group = self._per_group_breakdown()
        flags = self._generate_flags(dp_gap, eo_gap, di)

        return FairnessResult(
            sensitive_feature_name=self.sensitive_feature_name,
            demographic_parity_gap=float(dp_gap),
            equalized_odds_gap=float(eo_gap),
            disparate_impact=float(di),
            per_group_metrics=per_group,
            flags=flags,
        )

    # --------------------------------------------------------------- private

    def _disparate_impact(self) -> float:
        """
        Disparate Impact = (lowest group selection rate) / (highest group selection rate).
        Undefined (returns 1.0) when max selection rate is zero.
        """
        frame = MetricFrame(
            metrics=selection_rate,
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=self.sensitive_features,
        )
        rates = frame.by_group.values.astype(float)
        max_rate = rates.max()
        return (rates.min() / max_rate) if max_rate > 0 else 1.0

    def _per_group_breakdown(self) -> List[Dict[str, Any]]:
        """
        Build a per-group metric table using MetricFrame.
        """
        metrics = {
            "accuracy": accuracy_score,
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
            "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
            "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
        frame = MetricFrame(
            metrics=metrics,
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=self.sensitive_features,
        )
        rows = []
        for group_val, row in frame.by_group.iterrows():
            entry = {"group": str(group_val)}
            entry.update({k: round(float(v), 4) for k, v in row.items()})
            rows.append(entry)
        return rows

    def _generate_flags(
        self, dp_gap: float, eo_gap: float, di: float
    ) -> List[str]:
        flags: List[str] = []
        if abs(dp_gap) > self.dp_threshold:
            flags.append(
                f"Demographic parity gap {dp_gap:.3f} exceeds threshold {self.dp_threshold}. "
                f"Model selects positive outcomes at different rates across groups."
            )
        if abs(eo_gap) > self.eo_threshold:
            flags.append(
                f"Equalized odds gap {eo_gap:.3f} exceeds threshold {self.eo_threshold}. "
                f"TPR and/or FPR differ significantly across groups."
            )
        if di < self.di_threshold:
            flags.append(
                f"Disparate impact ratio {di:.3f} is below the EEOC 4/5ths threshold {self.di_threshold}. "
                f"May indicate adverse impact on a protected group."
            )
        return flags
