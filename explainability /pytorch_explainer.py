"""
mlens/explainability/pytorch_explainer.py
==========================================
Native PyTorch model explainability using SHAP DeepExplainer
and Captum (gradient-based attribution methods).

Supports:
  - DeepExplainer   → fast, approximation for deep nets
  - GradientExplainer → exact gradients (slower, more precise)
  - Integrated Gradients via Captum (optional)

Usage
-----
>>> from mlens.explainability.pytorch_explainer import PyTorchExplainer
>>> explainer = PyTorchExplainer(model, background_data)
>>> result = explainer.explain(X_test)
>>> result.top_features(n=10)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class PyTorchShapResult:
    """
    SHAP results for a PyTorch model.

    Attributes
    ----------
    shap_values : np.ndarray of shape (n_samples, n_features)
        SHAP values for every test observation.
    base_value : float
        Expected model output.
    feature_names : list of str
        Feature names aligned with shap_values columns.
    mean_abs_shap : np.ndarray
        Mean |SHAP| per feature.
    method : str
        Explainer method used ('deep' or 'gradient').
    """

    shap_values  : np.ndarray
    base_value   : float
    feature_names: List[str]
    method       : str = "deep"
    mean_abs_shap: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

    def top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return top-n features by mean |SHAP| value."""
        indices = np.argsort(self.mean_abs_shap)[::-1][:n]
        return [
            {
                "rank":          i + 1,
                "name":          self.feature_names[idx] if self.feature_names
                                 else f"feature_{idx}",
                "mean_abs_shap": round(float(self.mean_abs_shap[idx]), 6),
            }
            for i, idx in enumerate(indices)
        ]

    def local_explanation(self, sample_idx: int) -> List[Dict[str, Any]]:
        """Return per-feature SHAP contributions for one sample."""
        row   = self.shap_values[sample_idx]
        names = self.feature_names or [f"feature_{i}" for i in range(len(row))]
        return sorted(
            [{"feature": n, "shap_value": round(float(v), 6)}
             for n, v in zip(names, row)],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )


# ── Explainer ──────────────────────────────────────────────────────────────

class PyTorchExplainer:
    """
    Explain a PyTorch neural network using SHAP.

    Automatically selects:
      - DeepExplainer  for models with differentiable layers
      - GradientExplainer as fallback

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model in eval() mode.
    background_data : np.ndarray of shape (n_bg, n_features)
        Representative subset of training data for SHAP background.
        Keep 50–200 rows for performance.
    feature_names : list of str, optional
        Human-readable feature names.
    device : str
        Torch device string, e.g. 'cpu' or 'cuda' (default: 'cpu').
    method : str
        'auto' (default), 'deep', or 'gradient'.
    """

    def __init__(
        self,
        model          : Any,
        background_data: np.ndarray,
        feature_names  : Optional[List[str]] = None,
        device         : str = "cpu",
        method         : str = "auto",
    ) -> None:
        self.model           = model
        self.background_data = background_data
        self.feature_names   = feature_names
        self.device          = device
        self.method          = method
        self._explainer      = None

    # ---------------------------------------------------------------- public

    def explain(self, X_test: np.ndarray) -> PyTorchShapResult:
        """
        Compute SHAP values for X_test.

        Parameters
        ----------
        X_test : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        PyTorchShapResult
        """
        import torch
        import shap

        self.model.eval()

        # Convert to tensors
        bg_tensor   = torch.tensor(self.background_data, dtype=torch.float32).to(self.device)
        test_tensor = torch.tensor(X_test,               dtype=torch.float32).to(self.device)

        method_used = self.method

        try:
            if self.method in ("auto", "deep"):
                explainer   = shap.DeepExplainer(self.model, bg_tensor)
                shap_vals   = explainer.shap_values(test_tensor)
                base_value  = float(explainer.expected_value[0]
                              if isinstance(explainer.expected_value, (list, np.ndarray))
                              else explainer.expected_value)
                method_used = "deep"
            else:
                raise ValueError("Using gradient explainer")
        except Exception:
            # Fallback to GradientExplainer
            explainer   = shap.GradientExplainer(self.model, bg_tensor)
            shap_vals   = explainer.shap_values(test_tensor)
            base_value  = 0.0
            method_used = "gradient"

        shap_array = self._extract_shap_array(shap_vals)

        return PyTorchShapResult(
            shap_values  = shap_array,
            base_value   = base_value,
            feature_names= self.feature_names or
                           [f"feature_{i}" for i in range(shap_array.shape[1])],
            method       = method_used,
        )

    # --------------------------------------------------------------- captum

    def integrated_gradients(
        self,
        X_test     : np.ndarray,
        target_class: int = 0,
        n_steps    : int  = 50,
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients attributions via Captum.

        Parameters
        ----------
        X_test : np.ndarray
            Input samples to explain.
        target_class : int
            Output class index to attribute (default: 0).
        n_steps : int
            Number of IG approximation steps (default: 50).

        Returns
        -------
        dict with keys: 'attributions', 'delta', 'feature_names'
        """
        try:
            import torch
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError(
                "Captum is required for integrated gradients. "
                "Install with: pip install captum"
            )

        self.model.eval()
        ig      = IntegratedGradients(self.model)
        inputs  = torch.tensor(X_test,               dtype=torch.float32).to(self.device)
        baseline= torch.zeros_like(inputs)

        attributions, delta = ig.attribute(
            inputs,
            baselines        = baseline,
            target           = target_class,
            n_steps          = n_steps,
            return_convergence_delta=True,
        )

        attr_np = attributions.detach().cpu().numpy()
        return {
            "attributions": attr_np,
            "delta":        float(delta.mean().item()),
            "feature_names": self.feature_names or
                             [f"feature_{i}" for i in range(attr_np.shape[1])],
            "mean_abs_attr": np.abs(attr_np).mean(axis=0).tolist(),
        }

    # --------------------------------------------------------------- private

    @staticmethod
    def _extract_shap_array(raw: Any) -> np.ndarray:
        """Normalise SHAP output to (n_samples, n_features)."""
        if isinstance(raw, list):
            arr = np.array(raw[0])
        else:
            arr = np.array(raw)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr
