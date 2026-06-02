"""
mlens/drift/concept_drift.py
==============================
Concept drift detection on model predictions/errors over time.

Unlike feature drift (PSI/KS on input distributions),
concept drift tracks whether the MODEL BEHAVIOUR is changing:
  - Is the error rate increasing?
  - Is the prediction distribution shifting?
  - Has the relationship between features and labels changed?

Algorithms implemented
-----------------------
1. **ADWIN** (ADaptive WINdowing)
   - Maintains a sliding window of error observations
   - Splits window when mean error differs significantly between halves
   - Gold standard for online drift detection

2. **Page-Hinkley Test**
   - Sequential hypothesis test for mean shift detection
   - Lightweight, good for high-throughput streams
   - Detects when cumulative deviation exceeds threshold λ

3. **DDM** (Drift Detection Method)
   - Monitors warning and drift levels using statistical bounds
   - Based on Gama et al. (2004)

References
----------
- Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive Windowing" (2007)
- Page, "Continuous inspection schemes" (1954)
- Gama et al., "Learning with drift detection" (2004)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Result containers ──────────────────────────────────────────────────────

@dataclass
class ConceptDriftResult:
    """
    Results from concept drift detection over a prediction stream.

    Attributes
    ----------
    method : str
        Algorithm used ('adwin', 'page_hinkley', 'ddm').
    drift_detected : bool
        True if drift was confirmed.
    warning_detected : bool
        True if a warning level was reached (pre-drift signal).
    drift_indices : list of int
        Stream positions where drift was flagged.
    warning_indices : list of int
        Stream positions where warnings were raised.
    n_samples_processed : int
        Total observations processed.
    error_rate_before : float
        Mean error rate in the stable window before drift.
    error_rate_after : float
        Mean error rate after the detected change point.
    summary : str
        Plain-English summary of findings.
    """

    method               : str
    drift_detected       : bool
    warning_detected     : bool
    drift_indices        : List[int]
    warning_indices      : List[int]
    n_samples_processed  : int
    error_rate_before    : float
    error_rate_after     : float
    summary              : str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method":               self.method,
            "drift_detected":       self.drift_detected,
            "warning_detected":     self.warning_detected,
            "n_drift_points":       len(self.drift_indices),
            "n_warning_points":     len(self.warning_indices),
            "drift_indices":        self.drift_indices[:10],   # cap for JSON
            "n_samples_processed":  self.n_samples_processed,
            "error_rate_before":    round(self.error_rate_before, 4),
            "error_rate_after":     round(self.error_rate_after,  4),
            "summary":              self.summary,
        }


# ── ADWIN ──────────────────────────────────────────────────────────────────

class ADWIN:
    """
    ADaptive WINdowing drift detector.

    Maintains a variable-size window and detects when the
    mean of two sub-windows differs beyond a statistical threshold.

    Parameters
    ----------
    delta : float
        Confidence parameter (default: 0.002).
        Smaller → more sensitive, more false positives.
        Larger  → less sensitive, misses subtle drift.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta   = delta
        self._window : List[float] = []
        self._total  : float       = 0.0
        self.drift_detected  : bool = False
        self.warning_detected: bool = False

    def update(self, value: float) -> bool:
        """
        Add one observation and check for drift.

        Parameters
        ----------
        value : float
            Error value (0 = correct, 1 = incorrect prediction).

        Returns
        -------
        bool : True if drift was detected.
        """
        self._window.append(value)
        self._total += value
        self.drift_detected   = False
        self.warning_detected = False

        if len(self._window) < 2:
            return False

        self.drift_detected = self._detect_change()
        if self.drift_detected:
            # Reset window to the second half (post-change)
            mid = len(self._window) // 2
            self._window = self._window[mid:]
            self._total  = sum(self._window)

        return self.drift_detected

    def _detect_change(self) -> bool:
        """
        Test all possible split points in the window.
        Returns True if any split shows a significant mean difference.
        """
        n = len(self._window)
        cumsum = 0.0

        for i in range(1, n):
            cumsum += self._window[i - 1]
            n0, n1  = i, n - i
            mu0     = cumsum / n0
            mu1     = (self._total - cumsum) / n1

            # ADWIN bound
            m_inv   = (1.0 / n0) + (1.0 / n1)
            epsilon = math.sqrt(m_inv * math.log(2.0 * n / self.delta) / 2.0)

            if abs(mu0 - mu1) > epsilon:
                # Warning at half threshold
                self.warning_detected = abs(mu0 - mu1) > epsilon * 0.5
                return True
        return False

    @property
    def mean(self) -> float:
        """Current window mean."""
        return self._total / len(self._window) if self._window else 0.0

    @property
    def window_size(self) -> int:
        return len(self._window)


# ── Page-Hinkley ───────────────────────────────────────────────────────────

class PageHinkley:
    """
    Page-Hinkley test for detecting a shift in the mean of a stream.

    Flags drift when the cumulative deviation ΣT_t exceeds threshold λ.

    Parameters
    ----------
    threshold : float
        Detection threshold λ (default: 50.0).
        Higher → less sensitive but fewer false alarms.
    alpha : float
        Magnitude of acceptable change (default: 0.005).
    delta : float
        Minimum magnitude of a detected change (default: 0.005).
    """

    def __init__(
        self,
        threshold: float = 50.0,
        alpha    : float = 0.005,
        delta    : float = 0.005,
    ) -> None:
        self.threshold = threshold
        self.alpha     = alpha
        self.delta     = delta
        self._n        = 0
        self._sum      = 0.0
        self._x_mean   = 0.0
        self._ph_sum   = 0.0
        self._ph_min   = 0.0
        self.drift_detected  : bool = False
        self.warning_detected: bool = False

    def update(self, value: float) -> bool:
        """
        Add one observation and check for drift.

        Returns True if drift detected.
        """
        self._n     += 1
        self._sum   += value
        self._x_mean = self._sum / self._n

        self._ph_sum += value - self._x_mean - self.delta
        self._ph_min  = min(self._ph_min, self._ph_sum)

        ph_stat = self._ph_sum - self._ph_min

        self.drift_detected   = ph_stat > self.threshold
        self.warning_detected = ph_stat > self.threshold * 0.5

        if self.drift_detected:
            self._reset()

        return self.drift_detected

    def _reset(self) -> None:
        self._n      = 0
        self._sum    = 0.0
        self._x_mean = 0.0
        self._ph_sum = 0.0
        self._ph_min = 0.0


# ── DDM ────────────────────────────────────────────────────────────────────

class DDM:
    """
    Drift Detection Method (Gama et al., 2004).

    Tracks the running mean (p) and standard deviation (s) of errors.
    Raises warning when p + s > p_min + 2*s_min.
    Raises drift  when p + s > p_min + 3*s_min.

    Parameters
    ----------
    warning_level : float  (default: 2.0)
    drift_level   : float  (default: 3.0)
    min_samples   : int    Burn-in period before detection starts (default: 30)
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level  : float = 3.0,
        min_samples  : int   = 30,
    ) -> None:
        self.warning_level = warning_level
        self.drift_level   = drift_level
        self.min_samples   = min_samples
        self._reset()

    def _reset(self) -> None:
        self._n     = 0
        self._p     = 1.0
        self._s     = 0.0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self.drift_detected   = False
        self.warning_detected = False

    def update(self, value: float) -> bool:
        """
        Add one observation (0=correct, 1=error) and check for drift.
        Returns True if drift detected.
        """
        self._n += 1
        self._p  = self._p + (value - self._p) / self._n
        self._s  = math.sqrt(self._p * (1 - self._p) / self._n)

        self.drift_detected   = False
        self.warning_detected = False

        if self._n < self.min_samples:
            return False

        if self._p + self._s <= self._p_min + self.drift_level * self._s_min:
            self._p_min = min(self._p_min, self._p)
            self._s_min = min(self._s_min, self._s)

        if self._p + self._s > self._p_min + self.drift_level * self._s_min:
            self.drift_detected = True
            self._reset()
        elif self._p + self._s > self._p_min + self.warning_level * self._s_min:
            self.warning_detected = True

        return self.drift_detected


# ── High-level detector ────────────────────────────────────────────────────

class ConceptDriftDetector:
    """
    High-level concept drift detector that wraps ADWIN, Page-Hinkley, or DDM.

    Processes a stream of prediction errors and returns a ConceptDriftResult.

    Parameters
    ----------
    method : str
        'adwin' (default), 'page_hinkley', or 'ddm'.
    **kwargs
        Passed to the underlying algorithm constructor.
    """

    _ALGORITHMS = {
        "adwin":         ADWIN,
        "page_hinkley":  PageHinkley,
        "ddm":           DDM,
    }

    def __init__(self, method: str = "adwin", **kwargs) -> None:
        if method not in self._ALGORITHMS:
            raise ValueError(f"method must be one of {list(self._ALGORITHMS)}.")
        self.method    = method
        self.detector  = self._ALGORITHMS[method](**kwargs)

    # ---------------------------------------------------------------- public

    def detect(
        self,
        y_true: Any,
        y_pred: Any,
    ) -> ConceptDriftResult:
        """
        Process a full prediction stream and return drift diagnostics.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground-truth labels.
        y_pred : array-like of shape (n_samples,)
            Model predictions.

        Returns
        -------
        ConceptDriftResult
        """
        import numpy as np
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        errors = (y_true != y_pred).astype(float)

        drift_indices   : List[int] = []
        warning_indices : List[int] = []
        change_point    : int       = len(errors)

        for i, err in enumerate(errors):
            self.detector.update(err)
            if self.detector.drift_detected:
                drift_indices.append(i)
                if change_point == len(errors):
                    change_point = i
            if self.detector.warning_detected:
                warning_indices.append(i)

        # Compute error rates before / after first drift point
        if drift_indices:
            cp = drift_indices[0]
            error_before = float(errors[:cp].mean())   if cp > 0              else 0.0
            error_after  = float(errors[cp:].mean())   if cp < len(errors)    else 0.0
        else:
            mid          = len(errors) // 2
            error_before = float(errors[:mid].mean())
            error_after  = float(errors[mid:].mean())

        summary = self._build_summary(
            drift_indices, warning_indices, error_before, error_after
        )

        return ConceptDriftResult(
            method               = self.method,
            drift_detected       = len(drift_indices) > 0,
            warning_detected     = len(warning_indices) > 0,
            drift_indices        = drift_indices,
            warning_indices      = warning_indices,
            n_samples_processed  = len(errors),
            error_rate_before    = error_before,
            error_rate_after     = error_after,
            summary              = summary,
        )

    # --------------------------------------------------------------- private

    @staticmethod
    def _build_summary(
        drift   : List[int],
        warnings: List[int],
        before  : float,
        after   : float,
    ) -> str:
        if not drift and not warnings:
            return "✅ No concept drift detected. Model behaviour is stable."
        if drift:
            delta = after - before
            direction = "increased" if delta > 0 else "decreased"
            return (
                f"⚠️ Concept drift detected at {len(drift)} point(s). "
                f"Error rate {direction} from {before:.3f} → {after:.3f} "
                f"({delta:+.3f})."
            )
        return (
            f"⚠️ Warning level reached at {len(warnings)} point(s). "
            f"Monitor closely — potential drift emerging."
        )
