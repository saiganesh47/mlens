# mlens/recommendations/__init__.py
from mlens.recommendations.fairness_advisor import FairnessAdvisor, FairnessPlan
from mlens.recommendations.drift_advisor    import DriftAdvisor, DriftPlan
from mlens.recommendations.shap_advisor     import ShapAdvisor, ShapPlan

__all__ = [
    "FairnessAdvisor", "FairnessPlan",
    "DriftAdvisor",    "DriftPlan",
    "ShapAdvisor",     "ShapPlan",
]
