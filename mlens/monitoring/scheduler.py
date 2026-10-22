"""
mlens/monitoring/scheduler.py
================================
Schedule periodic MLens audits using a background thread.

Runs a full ModelAuditor pipeline on a fixed interval,
stores results, and fires alerts via AlertManager.

Usage
-----
>>> scheduler = AuditScheduler(
...     model=model,
...     X_train=X_train, X_test=X_test, y_test=y_test,
...     interval_seconds=3600,  # every hour
...     alert_manager=alert,
... )
>>> scheduler.start()
>>> scheduler.stop()
>>> scheduler.history()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


# ── Schedule entry ─────────────────────────────────────────────────────────

@dataclass
class ScheduleEntry:
    """
    A single completed audit run entry.

    Attributes
    ----------
    run_id : int
    timestamp : str
    runtime_seconds : float
    summary_lines : list of str
    drift_status : str | None
    fairness_flags : int
    error : str | None   Error message if the run failed.
    """
    run_id          : int
    timestamp       : str
    runtime_seconds : float
    summary_lines   : List[str]
    drift_status    : Optional[str]
    fairness_flags  : int
    error           : Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":          self.run_id,
            "timestamp":       self.timestamp,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "summary_lines":   self.summary_lines,
            "drift_status":    self.drift_status,
            "fairness_flags":  self.fairness_flags,
            "success":         self.success,
            "error":           self.error,
        }


# ── Scheduler ──────────────────────────────────────────────────────────────

class AuditScheduler:
    """
    Run MLens audits on a fixed interval in a background thread.

    Parameters
    ----------
    model : sklearn-compatible estimator
        The model to audit periodically.
    X_train : array-like
        Training data (drift reference).
    X_test : array-like
        Evaluation data.
    y_test : array-like
        Evaluation labels.
    interval_seconds : int
        How often to run the audit (default: 3600 = hourly).
    sensitive_features : array-like, optional
        Protected attribute for fairness checks.
    feature_names : list of str, optional
    model_name : str
    alert_manager : AlertManager, optional
        If provided, alerts are sent after each run.
    on_complete : callable, optional
        Callback(report, entry) called after each successful run.
    max_history : int
        Maximum number of run entries to keep (default: 100).
    """

    def __init__(
        self,
        model              : Any,
        X_train            : Any,
        X_test             : Any,
        y_test             : Any,
        interval_seconds   : int            = 3600,
        sensitive_features : Optional[Any]  = None,
        feature_names      : Optional[Any]  = None,
        model_name         : str            = "ScheduledModel",
        alert_manager      : Optional[Any]  = None,
        on_complete        : Optional[Callable] = None,
        max_history        : int            = 100,
    ) -> None:
        self.model              = model
        self.X_train            = X_train
        self.X_test             = X_test
        self.y_test             = y_test
        self.interval_seconds   = interval_seconds
        self.sensitive_features = sensitive_features
        self.feature_names      = feature_names
        self.model_name         = model_name
        self.alert_manager      = alert_manager
        self.on_complete        = on_complete
        self.max_history        = max_history

        self._history    : List[ScheduleEntry] = []
        self._run_id     : int   = 0
        self._thread     : Optional[threading.Thread] = None
        self._stop_event : threading.Event = threading.Event()
        self._lock       : threading.Lock  = threading.Lock()

    # ---------------------------------------------------------------- public

    def start(self) -> None:
        """Start the background audit scheduler."""
        if self._thread and self._thread.is_alive():
            print(f"[MLens Scheduler] Already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"mlens-scheduler-{self.model_name}",
        )
        self._thread.start()
        print(
            f"[MLens Scheduler] Started — auditing '{self.model_name}' "
            f"every {self.interval_seconds}s."
        )

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        print(f"[MLens Scheduler] Stopped.")

    def run_now(self) -> ScheduleEntry:
        """Trigger an immediate audit run (blocking)."""
        return self._audit_once()

    def history(self, n: int = 20) -> List[ScheduleEntry]:
        """Return the last n completed audit entries."""
        with self._lock:
            return list(self._history[-n:])

    def latest(self) -> Optional[ScheduleEntry]:
        """Return the most recent completed entry."""
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --------------------------------------------------------------- private

    def _run_loop(self) -> None:
        """Background thread: audit → sleep → audit → …"""
        while not self._stop_event.is_set():
            self._audit_once()
            self._stop_event.wait(timeout=self.interval_seconds)

    def _audit_once(self) -> ScheduleEntry:
        """Run one full audit and store the result."""
        from mlens.auditor import ModelAuditor

        self._run_id += 1
        run_id = self._run_id
        ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0     = time.perf_counter()

        print(f"[MLens Scheduler] Run #{run_id} starting …")

        try:
            auditor = ModelAuditor(
                model              = self.model,
                X_train            = self.X_train,
                X_test             = self.X_test,
                y_test             = self.y_test,
                sensitive_features = self.sensitive_features,
                feature_names      = self.feature_names,
                model_name         = self.model_name,
            )
            report = auditor.run()

            entry = ScheduleEntry(
                run_id          = run_id,
                timestamp       = ts,
                runtime_seconds = time.perf_counter() - t0,
                summary_lines   = report.summary_lines,
                drift_status    = (report.drift_result.overall_status
                                   if report.drift_result else None),
                fairness_flags  = (len(report.fairness_result.flags)
                                   if report.fairness_result else 0),
            )

            if self.alert_manager:
                self.alert_manager.check_and_notify(report)

            if self.on_complete:
                try:
                    self.on_complete(report, entry)
                except Exception as cb_exc:
                    print(f"[MLens Scheduler] Callback error: {cb_exc}")

            print(
                f"[MLens Scheduler] Run #{run_id} complete "
                f"({entry.runtime_seconds:.2f}s) — "
                f"drift: {entry.drift_status or 'n/a'}, "
                f"fairness flags: {entry.fairness_flags}"
            )

        except Exception as exc:
            entry = ScheduleEntry(
                run_id          = run_id,
                timestamp       = ts,
                runtime_seconds = time.perf_counter() - t0,
                summary_lines   = [],
                drift_status    = None,
                fairness_flags  = 0,
                error           = str(exc),
            )
            print(f"[MLens Scheduler] Run #{run_id} failed: {exc}")

        with self._lock:
            self._history.append(entry)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]

        return entry
