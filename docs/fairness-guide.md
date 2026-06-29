# Fairness Guide

How MLens evaluates fairness, and what each metric actually means.

## Why Fairness Matters

A model can have 95% accuracy and still treat one group of people
systematically worse than another. Accuracy alone never reveals this —
you have to look at outcomes *broken down by group*.

## The Three Metrics

### 1. Demographic Parity Gap

**Question it answers:** Does the model approve/select people at the
same rate across groups?

```
Demographic Parity Gap = |P(ŷ=1 | group A) − P(ŷ=1 | group B)|
```

- **Threshold:** 0.10 (configurable)
- **Limitation:** Doesn't account for whether the underlying base rates
  actually differ between groups — a gap isn't always unjust, but it's
  always worth investigating.

### 2. Equalized Odds Gap

**Question it answers:** Among people who *should* get a positive outcome,
does the model find them at the same rate across groups? (And same for
people who shouldn't.)

```
Equalized Odds Gap = max(|TPR_A − TPR_B|, |FPR_A − FPR_B|)
```

- **Threshold:** 0.10 (configurable)
- **Stronger guarantee** than demographic parity — it accounts for the
  true labels, not just the predictions.

### 3. Disparate Impact Ratio

**Question it answers:** The EEOC's legal "4/5ths rule" for adverse impact
in employment decisions.

```
Disparate Impact = min(selection_rate) / max(selection_rate)
```

- **Threshold:** 0.80 (a ratio below this is a legal red flag in US
  employment law)
- **Use case:** Hiring, lending, and other regulated decision domains.

## Per-Group Breakdown

Beyond the three headline metrics, MLens also reports for each group:

| Metric | What it tells you |
|---|---|
| Accuracy | Overall correctness within the group |
| Precision | Of positive predictions, how many were correct |
| Recall | Of actual positives, how many were caught |
| F1 | Harmonic mean of precision and recall |
| Selection Rate | % of the group receiving a positive prediction |
| False Positive Rate | % of negatives incorrectly flagged positive |

If two groups have similar accuracy but very different recall or FPR,
that's often the real story — accuracy alone hides it.

## What To Do When a Flag Fires

1. **Don't panic** — a flag means "investigate", not "the model is illegal."
2. **Check the base rates.** If group A genuinely has a different rate of
   the outcome in the ground truth, some gap is expected — but should be
   explainable and defensible.
3. **Look at the per-group breakdown** to see *which* metric drove the gap.
4. **Consider mitigation:** reweighting training data, fairness-constrained
   training (e.g. `fairlearn.reductions`), or post-processing thresholds.
5. **Re-run the audit** after any change to confirm the gap closed.

## Configuring Thresholds

```python
from mlens.fairness.fairness_metrics import FairnessEvaluator

evaluator = FairnessEvaluator(
    y_true, y_pred, sensitive_features,
    dp_threshold=0.05,   # stricter than default 0.10
    eo_threshold=0.05,
    di_threshold=0.90,   # stricter than default 0.80
)
```

## References

- Hardt, Price & Srebro, *Equality of Opportunity in Supervised Learning* (NeurIPS 2016)
- Bird et al., *Fairlearn: A toolkit for assessing and improving fairness in AI* (2020)
- EEOC *Uniform Guidelines on Employee Selection Procedures* (1978)
