"""
mlens/drift/drift_detector.py
===============================
Feature-level data drift detection using two complementary methods:

1. **Population Stability Index (PSI)**
   - Industry standard in credit risk & financial ML
   - PSI < 0.10  → stable
   - 0.10 ≤ PSI < 0.25 → moderate shift (monitor)
   - PSI ≥ 0.25  → significant drift

2. **Kolmogorov-Smirnov (KS) Test**
   - Non-parametric two-sample test on the empirical distributions
   - Flags features where p-value < alpha (default 0.05)

A feature is flagged as "drifted" if EITHER method detects a shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PSI_STABLE    = 0.10
PSI_MODERATE  = 0.25
PSI_N_BINS    = 10
KS_ALPHA      = 0.05


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """
    Aggregated drift diagnostics for all features.

    Attributes
    ----------
    feature_results : list of dict
        Per-feature results with keys:
        feature, psi, psi_status, ks_statistic, ks_pvalue, drifted.
    n_drifted : int
        Count of features flagged as drifted.
    overall_status : str
        'stable' | 'moderate' | 'significant'
    """

    feature_results: List[Dict[str, Any]]
    n_drifted: int
    overall_status: str

    def drifted_features(self) -> List[str]:
        """Return names of features with detected drift."""
        return [f["feature"] for f in self.feature_results if f["drifted"]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "n_drifted": self.n_drifted,
            "drifted_features": self.drifted_features(),
            "details": self.feature_results,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Detect feature-level distribution shift between a reference and
    a production dataset.

    Parameters
    ----------
    reference : np.ndarray of shape (n_ref, n_features)
        Training / reference distribution.
    feature_names : list of str, optional
        Human-readable feature names.
    psi_bins : int
        Number of equal-frequency bins used for PSI calculation.
    ks_alpha : float
        Significance level for the KS test (default: 0.05).
    """

    def __init__(
        self,
        reference: np.ndarray,
        feature_names: Optional[List[str]] = None,
        psi_bins: int = PSI_N_BINS,
        ks_alpha: float = KS_ALPHA,
    ) -> None:
        self.reference = reference
        self.feature_names = feature_names or [f"feature_{i}" for i in range(reference.shape[1])]
        self.psi_bins = psi_bins
        self.ks_alpha = ks_alpha

    # ---------------------------------------------------------------- public

    def detect(self, production: np.ndarray) -> DriftResult:
        """
        Compare production data against the reference distribution.

        Parameters
        ----------
        production : np.ndarray of shape (n_prod, n_features)

        Returns
        -------
        DriftResult
        """
        n_features = self.reference.shape[1]
        results: List[Dict[str, Any]] = []

        for i in range(n_features):
            ref_col  = self.reference[:, i].astype(float)
            prod_col = production[:, i].astype(float)
            name     = self.feature_names[i]

            psi_val    = self._compute_psi(ref_col, prod_col)
            psi_status = self._psi_label(psi_val)
            ks_stat, ks_pval = stats.ks_2samp(ref_col, prod_col)

            drifted = (psi_val >= PSI_MODERATE) or (ks_pval < self.ks_alpha)

            results.append({
                "feature":       name,
                "psi":           round(float(psi_val), 5),
                "psi_status":    psi_status,
                "ks_statistic":  round(float(ks_stat), 5),
                "ks_pvalue":     round(float(ks_pval), 5),
                "drifted":       drifted,
            })

        n_drifted = sum(1 for r in results if r["drifted"])
        max_psi   = max(r["psi"] for r in results) if results else 0.0
        overall   = self._overall_status(max_psi, n_drifted)

        return DriftResult(
            feature_results=results,
            n_drifted=n_drifted,
            overall_status=overall,
        )

    # --------------------------------------------------------------- private

    def _compute_psi(
        self, reference: np.ndarray, production: np.ndarray
    ) -> float:
        """
        Population Stability Index using equal-frequency binning on the
        reference distribution.

        PSI = Σ (actual% - expected%) × ln(actual% / expected%)

        A small epsilon avoids log(0) errors.
        """
        # Build bins from reference quantiles
        quantiles = np.linspace(0, 100, self.psi_bins + 1)
        bin_edges = np.unique(np.percentile(reference, quantiles))

        if len(bin_edges) < 2:
            # Constant feature — no drift possible
            return 0.0

        ref_counts,  _ = np.histogram(reference,  bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # Normalize to proportions; add epsilon to avoid division-by-zero
        eps = 1e-8
        ref_pct  = ref_counts  / (len(reference)  + eps) + eps
        prod_pct = prod_counts / (len(production) + eps) + eps

        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return float(max(psi, 0.0))

    @staticmethod
    def _psi_label(psi: float) -> str:
        if psi < PSI_STABLE:
            return "stable"
        if psi < PSI_MODERATE:
            return "moderate"
        return "significant"

    @staticmethod
    def _overall_status(max_psi: float, n_drifted: int) -> str:
        if n_drifted == 0:
            return "stable"
        if max_psi >= PSI_MODERATE:
            return "significant"
        return "moderate"
