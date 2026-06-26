# Python API Reference

## `mlens.auditor.ModelAuditor`

The main orchestrator. Runs SHAP, fairness, and drift analysis on a trained model.

```python
ModelAuditor(
    model,
    X_train,
    X_test,
    y_test,
    sensitive_features=None,
    feature_names=None,
    model_name=None,
    shap_background_samples=100,
    run_drift=True,
    run_fairness=True,
    run_shap=True,
)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | estimator | Any sklearn-compatible, XGBoost, or LightGBM model |
| `X_train` | array-like | Training data — used for drift reference and SHAP background |
| `X_test` | array-like | Held-out evaluation data |
| `y_test` | array-like | Ground-truth labels for `X_test` |
| `sensitive_features` | array-like, optional | Protected attribute column (e.g. gender) |
| `feature_names` | list[str], optional | Auto-inferred if `X_train` is a DataFrame |
| `model_name` | str, optional | Display name in the report |
| `shap_background_samples` | int | Rows sampled for the SHAP background (default 100) |

### `.run() → AuditReport`

Executes the full pipeline and returns a populated `AuditReport`.

---

## `mlens.auditor.AuditReport`

Returned by `ModelAuditor.run()`.

| Attribute | Type | Description |
|---|---|---|
| `model_name` | str | Model display name |
| `runtime_seconds` | float | Total audit runtime |
| `shap_result` | `ShapResult` \| None | SHAP findings |
| `fairness_result` | `FairnessResult` \| None | Fairness findings |
| `drift_result` | `DriftResult` \| None | Drift findings |
| `summary_lines` | list[str] | Plain-English findings |

### `.save(path) → Path`
Saves an HTML (or PDF, if path ends in `.pdf`) report.

### `.to_dict() → dict`
JSON-serialisable summary of all results.

---

## `mlens.explainability.shap_analyzer.ShapAnalyzer`

```python
ShapAnalyzer(model, background_data, feature_names=None)
```

### `.explain(X_test) → ShapResult`

`ShapResult` methods:
- `.top_features(n=10)` — top-n features by mean |SHAP|
- `.local_explanation(sample_idx)` — per-feature contribution for one row

---

## `mlens.fairness.fairness_metrics.FairnessEvaluator`

```python
FairnessEvaluator(
    y_true, y_pred, sensitive_features,
    sensitive_feature_name="sensitive_feature",
    dp_threshold=0.10, eo_threshold=0.10, di_threshold=0.80,
)
```

### `.evaluate() → FairnessResult`

`FairnessResult` attributes:
- `demographic_parity_gap`, `equalized_odds_gap`, `disparate_impact`
- `per_group_metrics` — list of dicts per protected group
- `flags` — list of triggered warning strings
- `.is_fair` — bool, True if no flags raised

---

## `mlens.drift.drift_detector.DriftDetector`

```python
DriftDetector(reference, feature_names=None, psi_bins=10, ks_alpha=0.05)
```

### `.detect(production) → DriftResult`

`DriftResult` attributes:
- `feature_results` — list of per-feature PSI/KS results
- `overall_status` — `'stable'` | `'moderate'` | `'significant'`
- `.drifted_features()` — list of feature names flagged

---

## `mlens.drift.concept_drift.ConceptDriftDetector`

```python
ConceptDriftDetector(method="adwin")  # or "page_hinkley", "ddm"
```

### `.detect(y_true, y_pred) → ConceptDriftResult`

Tracks model error rate over a prediction stream to catch
behavioural drift that feature-level drift detection misses.

---

## `mlens.explainability.pytorch_explainer.PyTorchExplainer`

```python
PyTorchExplainer(model, background_data, feature_names=None, device="cpu")
```

### `.explain(X_test) → PyTorchShapResult`
### `.integrated_gradients(X_test, target_class=0) → dict`

---

## `mlens.integrations`

```python
from mlens.integrations import MLflowTracker, WandbTracker

MLflowTracker(experiment_name="mlens-audits").log(report)
WandbTracker(project="mlens-audits").log(report)
```
