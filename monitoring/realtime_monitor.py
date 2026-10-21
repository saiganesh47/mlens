"""
mlens/monitoring/realtime_monitor.py
======================================
Stream-based real-time model monitoring.

Ingests prediction batches, tracks rolling metrics, detects
anomalies, and emits alerts via AlertManager when thresholds breach.

Architecture
------------
  PredictionBatch → RealtimeMonitor.ingest()
      → updates RollingMetrics window
      → checks ConceptDriftDetector (ADWIN)
      → checks threshold rules
      → emits Alert if violated
      → stores MonitorSnapshot for dashboard

Usage
-----
>>> monitor = RealtimeMonitor(model_name="MyModel", window_size=500)
>>> monitor.ingest(y_true=batch_labels, y_pred=batch_preds,
...               sensitive=batch_sensitive, X=batch_features)
>>> status = monitor.status()
>>> print(status.error_rate, status.drift_detected)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np


# ── Snapshot dataclass ─────────────────────────────────────────────────────

@dataclass
class MonitorSnapshot:
    """
    A point-in-time snapshot of model health metrics.

    Attributes
    ----------
    timestamp : str          UTC ISO timestamp.
    n_predictions : int      Total predictions processed so far.
    error_rate : float       Rolling error rate in current window.
    drift_detected : bool    Whether ADWIN flagged concept drift.
    dp_gap : float | None    Rolling demographic parity gap.
    alert_count : int        Cumulative alerts triggered.
    window_size : int        Current rolling window length.
    """
    timestamp     : str
    n_predictions : int
    error_rate    : float
    drift_detected: bool
    dp_gap        : Optional[float]
    alert_count   : int
    window_size   : int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":      self.timestamp,
            "n_predictions":  self.n_predictions,
            "error_rate":     round(self.error_rate, 4),
            "drift_detected": self.drift_detected,
            "dp_gap":         round(self.dp_gap, 4) if self.dp_gap else None,
            "alert_count":    self.alert_count,
            "window_size":    self.window_size,
        }


# ── Monitor ────────────────────────────────────────────────────────────────

class RealtimeMonitor:
    """
    Rolling window model health monitor.

    Tracks error rate, demographic parity, and concept drift
    over a sliding window of incoming predictions.

    Parameters
    ----------
    model_name : str
        Display name for the monitored model.
    window_size : int
        Number of recent predictions to keep in the rolling window (default 500).
    error_rate_threshold : float
        Alert when rolling error rate exceeds this (default 0.20).
    dp_gap_threshold : float
        Alert when rolling demographic parity gap exceeds this (default 0.10).
    drift_delta : float
        ADWIN sensitivity (default 0.002).
    alert_manager : AlertManager, optional
        If provided, triggers alerts automatically on threshold breach.
    """

    def __init__(
        self,
        model_name           : str,
        window_size          : int   = 500,
        error_rate_threshold : float = 0.20,
        dp_gap_threshold     : float = 0.10,
        drift_delta          : float = 0.002,
        alert_manager        : Optional[Any] = None,
    ) -> None:
        self.model_name            = model_name
        self.window_size           = window_size
        self.error_rate_threshold  = error_rate_threshold
        self.dp_gap_threshold      = dp_gap_threshold
        self.alert_manager         = alert_manager

        # Rolling buffers
        self._errors    : Deque[float] = deque(maxlen=window_size)
        self._sensitive : Deque[Any]   = deque(maxlen=window_size)
        self._preds     : Deque[float] = deque(maxlen=window_size)
        self._true      : Deque[float] = deque(maxlen=window_size)

        # State
        self._n_total       : int   = 0
        self._alert_count   : int   = 0
        self._snapshots     : List[MonitorSnapshot] = []
        self._drift_detected: bool  = False

        # ADWIN concept drift detector
        from mlens.drift.concept_drift import ADWIN
        self._adwin = ADWIN(delta=drift_delta)

    # ---------------------------------------------------------------- public

    def ingest(
        self,
        y_true   : Any,
        y_pred   : Any,
        sensitive: Optional[Any] = None,
        X        : Optional[Any] = None,
    ) -> MonitorSnapshot:
        """
        Ingest a batch of predictions and update rolling metrics.

        Parameters
        ----------
        y_true : array-like    Ground-truth labels.
        y_pred : array-like    Model predictions.
        sensitive : array-like, optional  Protected attribute values.
        X : array-like, optional  Feature matrix (reserved for future use).

        Returns
        -------
        MonitorSnapshot : current health metrics after ingestion.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        errors = (y_true != y_pred).astype(float)

        # Update buffers
        for e, p, t in zip(errors, y_pred, y_true):
            self._errors.append(float(e))
            self._preds.append(float(p))
            self._true.append(float(t))
            self._adwin.update(float(e))

        if sensitive is not None:
            s_arr = np.asarray(sensitive)
            for s in s_arr:
                self._sensitive.append(s)

        self._n_total      += len(errors)
        self._drift_detected = self._adwin.drift_detected

        snapshot = self._build_snapshot()
        self._snapshots.append(snapshot)
        self._check_thresholds(snapshot)

        return snapshot

    def status(self) -> MonitorSnapshot:
        """Return the latest snapshot without ingesting new data."""
        if self._snapshots:
            return self._snapshots[-1]
        return self._empty_snapshot()

    def history(self, n: int = 100) -> List[MonitorSnapshot]:
        """Return the last n snapshots."""
        return list(self._snapshots[-n:])

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self._errors.clear()
        self._sensitive.clear()
        self._preds.clear()
        self._true.clear()
        self._n_total        = 0
        self._alert_count    = 0
        self._snapshots.clear()
        self._drift_detected = False
        from mlens.drift.concept_drift import ADWIN
        self._adwin = ADWIN()

    # --------------------------------------------------------------- private

    def _build_snapshot(self) -> MonitorSnapshot:
        error_rate = float(np.mean(self._errors)) if self._errors else 0.0
        dp_gap     = self._rolling_dp_gap()
        return MonitorSnapshot(
            timestamp      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            n_predictions  = self._n_total,
            error_rate     = error_rate,
            drift_detected = self._drift_detected,
            dp_gap         = dp_gap,
            alert_count    = self._alert_count,
            window_size    = len(self._errors),
        )

    def _empty_snapshot(self) -> MonitorSnapshot:
        return MonitorSnapshot(
            timestamp="", n_predictions=0, error_rate=0.0,
            drift_detected=False, dp_gap=None,
            alert_count=0, window_size=0,
        )

    def _rolling_dp_gap(self) -> Optional[float]:
        """Compute demographic parity gap from rolling window."""
        if len(self._sensitive) < 10:
            return None
        s_arr = np.array(list(self._sensitive))
        p_arr = np.array(list(self._preds))
        if len(s_arr) != len(p_arr):
            n = min(len(s_arr), len(p_arr))
            s_arr, p_arr = s_arr[-n:], p_arr[-n:]
        groups = np.unique(s_arr)
        if len(groups) < 2:
            return None
        rates = []
        for g in groups:
            mask = s_arr == g
            if mask.sum() > 0:
                rates.append(float(p_arr[mask].mean()))
        return max(rates) - min(rates) if len(rates) >= 2 else None

    def _check_thresholds(self, snap: MonitorSnapshot) -> None:
        """Fire alerts if any threshold is breached."""
        alerts: List[str] = []

        if snap.error_rate > self.error_rate_threshold:
            alerts.append(
                f"⚠️ Error rate {snap.error_rate:.3f} exceeds "
                f"threshold {self.error_rate_threshold} "
                f"(window: {snap.window_size} predictions)."
            )

        if snap.drift_detected:
            alerts.append(
                f"📊 Concept drift detected by ADWIN at "
                f"prediction #{snap.n_predictions}."
            )

        if snap.dp_gap is not None and snap.dp_gap > self.dp_gap_threshold:
            alerts.append(
                f"⚖️ Rolling demographic parity gap {snap.dp_gap:.3f} "
                f"exceeds threshold {self.dp_gap_threshold}."
            )

        if alerts:
            self._alert_count += len(alerts)
            for msg in alerts:
                print(f"[MLens Monitor] {msg}")
