# Getting Started with MLens

This guide gets you from `pip install` to your first audit report in under 5 minutes.

## 1. Install

```bash
git clone https://github.com/saiganesh47/mlens.git
cd mlens
pip install -r requirements.txt
```

## 2. Run Your First Audit (Python)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from mlens import ModelAuditor

# Your existing model + data
X, y = make_classification(n_samples=1000, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier().fit(X_train, y_train)

# Run the audit
auditor = ModelAuditor(model, X_train, X_test, y_test)
report = auditor.run()

# Read findings
for line in report.summary_lines:
    print(line)

# Save a shareable report
report.save("audit_report.html")
```

## 3. Add Fairness Checks

If you have a protected attribute (gender, race, age group, etc.), pass it in:

```python
auditor = ModelAuditor(
    model, X_train, X_test, y_test,
    sensitive_features=df_test["gender"],
)
report = auditor.run()
print(report.fairness_result.flags)
```

## 4. Run It From the Terminal

```bash
mlens audit model.pkl X_test.csv y_test.csv --sensitive gender.csv
```

## 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`, upload your model and CSVs, click **Run Audit**.

## 6. Or Run the REST API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Next Steps

- [CLI Reference](cli-reference.md) — every command and flag
- [API Reference](api-reference.md) — Python classes and methods
- [Fairness Guide](fairness-guide.md) — how the fairness metrics work
- [Jupyter Tutorial](../notebooks/mlens_tutorial.ipynb) — full walkthrough notebook
