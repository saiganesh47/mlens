"""
mlens/automl/hyperparameter_tuner.py
======================================
Auto-tune model hyperparameters with an optional fairness constraint.

Supports two tuning strategies:
  1. Accuracy-only  — standard cross-validated grid/random search
  2. Fairness-aware — optimises accuracy subject to a demographic
                      parity gap ≤ threshold (multi-objective)

Usage
-----
>>> from mlens.automl.hyperparameter_tuner import HyperparameterTuner
>>> tuner = HyperparameterTuner(
...     model=RandomForestClassifier(),
...     X_train=X_train, y_train=y_train,
...     sensitive_train=s_train,
...     fairness_constraint=0.10,
... )
>>> result = tuner.tune()
>>> result.best_model
>>> result.best_params
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class TuningResult:
    """
    Result from a hyperparameter tuning run.

    Attributes
    ----------
    best_model : fitted estimator
        Model fitted with the best found hyperparameters.
    best_params : dict
        Best hyperparameter values.
    best_score : float
        Cross-validated accuracy/F1 with best params.
    fairness_gap : float | None
        Demographic parity gap of the best model (None if not computed).
    n_trials : int
        Number of hyperparameter combinations tried.
    runtime_seconds : float
        Total tuning time.
    search_history : list of dict
        Per-trial results for plotting convergence.
    """
    best_model      : Any
    best_params     : Dict[str, Any]
    best_score      : float
    fairness_gap    : Optional[float]
    n_trials        : int
    runtime_seconds : float
    search_history  : List[Dict[str, Any]]

    def summary(self) -> str:
        lines = [
            f"Best score         : {self.best_score:.4f}",
            f"Best params        : {self.best_params}",
            f"Fairness gap       : {self.fairness_gap:.4f}" if self.fairness_gap is not None
                                                            else "Fairness gap : N/A",
            f"Trials run         : {self.n_trials}",
            f"Runtime            : {self.runtime_seconds:.2f}s",
        ]
        return "\n".join(lines)


# ── Tuner ──────────────────────────────────────────────────────────────────

class HyperparameterTuner:
    """
    Random search with optional fairness constraint.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model to tune.
    X_train : array-like
    y_train : array-like
    param_grid : dict of {param: list of values}, optional
        Search space. If None, uses a built-in grid for common models.
    sensitive_train : array-like, optional
        Protected attribute aligned with X_train/y_train.
        Required for fairness-aware tuning.
    fairness_constraint : float, optional
        Maximum allowed demographic parity gap (default: None = no constraint).
    n_trials : int
        Number of random hyperparameter combinations to try (default: 20).
    cv : int
        Cross-validation folds (default: 5).
    scoring : str
        Sklearn scoring string (default: 'f1_weighted').
    random_state : int
    """

    # Built-in search spaces for common model types
    _DEFAULT_GRIDS: Dict[str, Dict] = {
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth":    [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"],
        },
        "GradientBoostingClassifier": {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth":     [3, 4, 5, 6],
            "subsample":     [0.7, 0.8, 1.0],
        },
        "XGBClassifier": {
            "n_estimators":      [100, 200, 300],
            "learning_rate":     [0.01, 0.05, 0.1],
            "max_depth":         [3, 4, 5, 6],
            "subsample":         [0.7, 0.8, 1.0],
            "colsample_bytree":  [0.7, 0.8, 1.0],
        },
        "LogisticRegression": {
            "C":        [0.001, 0.01, 0.1, 1.0, 10.0],
            "penalty":  ["l1", "l2"],
            "solver":   ["liblinear", "saga"],
            "max_iter": [500, 1000],
        },
        "LGBMClassifier": {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves":    [31, 63, 127],
            "subsample":     [0.7, 0.8, 1.0],
        },
    }

    def __init__(
        self,
        model               : Any,
        X_train             : Any,
        y_train             : Any,
        param_grid          : Optional[Dict] = None,
        sensitive_train     : Optional[Any]  = None,
        fairness_constraint : Optional[float]= None,
        n_trials            : int            = 20,
        cv                  : int            = 5,
        scoring             : str            = "f1_weighted",
        random_state        : int            = 42,
    ) -> None:
        self.model               = model
        self.X_train             = np.asarray(X_train)
        self.y_train             = np.asarray(y_train)
        self.param_grid          = param_grid or self._infer_grid()
        self.sensitive_train     = (np.asarray(sensitive_train)
                                    if sensitive_train is not None else None)
        self.fairness_constraint = fairness_constraint
        self.n_trials            = n_trials
        self.cv                  = cv
        self.scoring             = scoring
        self.rng                 = np.random.default_rng(random_state)

    # ---------------------------------------------------------------- public

    def tune(self) -> TuningResult:
        """
        Run random search and return the best hyperparameter configuration.

        If fairness_constraint is set, only configurations whose
        demographic parity gap ≤ constraint are considered.

        Returns
        -------
        TuningResult
        """
        t0      = time.perf_counter()
        history : List[Dict] = []
        best_score  = -np.inf
        best_params : Dict  = {}
        best_model  : Any   = None
        best_gap    : Optional[float] = None

        param_samples = self._sample_params(self.n_trials)
        print(f"\n[MLens Tuner] Tuning {type(self.model).__name__} "
              f"({self.n_trials} trials) …\n")

        for i, params in enumerate(param_samples):
            try:
                model_copy = self._clone_with_params(params)
                score      = self._cross_val(model_copy)
                gap        = None

                if self.fairness_constraint is not None and self.sensitive_train is not None:
                    gap = self._estimate_fairness_gap(model_copy)
                    if gap > self.fairness_constraint:
                        history.append({
                            "trial": i + 1, "score": score,
                            "fairness_gap": gap, "accepted": False,
                            "params": params,
                        })
                        continue  # skip — violates fairness constraint

                if score > best_score:
                    best_score  = score
                    best_params = params
                    best_gap    = gap
                    # Refit on full training data
                    best_model  = self._clone_with_params(params)
                    best_model.fit(self.X_train, self.y_train)

                history.append({
                    "trial": i + 1, "score": score,
                    "fairness_gap": gap, "accepted": True,
                    "params": params,
                })
                print(f"  Trial {i+1:>3}/{self.n_trials} — "
                      f"score={score:.4f}"
                      + (f"  gap={gap:.4f}" if gap is not None else "")
                      + (" ✓ best" if score == best_score else ""))

            except Exception as exc:
                warnings.warn(f"Trial {i+1} failed: {exc}")
                continue

        if best_model is None:
            warnings.warn(
                "All trials violated the fairness constraint. "
                "Returning best unconstrained model."
            )
            best_model, best_params, best_score, best_gap = \
                self._unconstrained_best(param_samples)

        elapsed = time.perf_counter() - t0
        print(f"\n[MLens Tuner] Best score: {best_score:.4f} "
              f"in {elapsed:.2f}s")

        return TuningResult(
            best_model      = best_model,
            best_params     = best_params,
            best_score      = best_score,
            fairness_gap    = best_gap,
            n_trials        = self.n_trials,
            runtime_seconds = elapsed,
            search_history  = history,
        )

    # --------------------------------------------------------------- private

    def _infer_grid(self) -> Dict:
        name = type(self.model).__name__
        return self._DEFAULT_GRIDS.get(name, {
            "n_estimators": [100, 200],
            "max_depth":    [None, 5, 10],
        })

    def _sample_params(self, n: int) -> List[Dict]:
        if not self.param_grid:
            return [{}] * n
        samples = []
        for _ in range(n):
            sample = {
                k: self.rng.choice(v).item()
                   if isinstance(v[0], (int, float, np.integer, np.floating))
                   else str(self.rng.choice(v))
                for k, v in self.param_grid.items()
            }
            samples.append(sample)
        return samples

    def _clone_with_params(self, params: Dict) -> Any:
        from sklearn.base import clone
        m = clone(self.model)
        valid = {k: v for k, v in params.items()
                 if k in m.get_params()}
        m.set_params(**valid)
        return m

    def _cross_val(self, model: Any) -> float:
        cv     = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=cv, scoring=self.scoring, error_score=0.0,
        )
        return float(scores.mean())

    def _estimate_fairness_gap(self, model: Any) -> float:
        """Estimate demographic parity gap via cross-val predictions."""
        from sklearn.model_selection import cross_val_predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(
                model, self.X_train, self.y_train,
                cv=min(self.cv, 3),
            )
        groups      = np.unique(self.sensitive_train)
        rates       = []
        for g in groups:
            mask = self.sensitive_train == g
            if mask.sum() > 0:
                rates.append(float(y_pred[mask].mean()))
        return max(rates) - min(rates) if len(rates) >= 2 else 0.0

    def _unconstrained_best(
        self, param_samples: List[Dict]
    ) -> Tuple[Any, Dict, float, None]:
        best_score  = -np.inf
        best_params = {}
        best_model  = self._clone_with_params({})
        for params in param_samples:
            try:
                m = self._clone_with_params(params)
                s = self._cross_val(m)
                if s > best_score:
                    best_score  = s
                    best_params = params
                    best_model  = m
            except Exception:
                continue
        best_model.fit(self.X_train, self.y_train)
        return best_model, best_params, best_score, None
