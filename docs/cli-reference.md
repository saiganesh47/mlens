# CLI Reference

Full reference for the `mlens` command-line tool.

## Installation

```bash
pip install -e .
```

This registers `mlens` as a system command.

## Commands

### `mlens audit`

Run a full ML audit from the terminal.

```bash
mlens audit MODEL X_TEST Y_TEST [OPTIONS]
```

**Positional arguments:**

| Argument | Description |
|---|---|
| `MODEL` | Path to a serialised model (`.pkl` or `.joblib`) |
| `X_TEST` | Path to test features CSV |
| `Y_TEST` | Path to test labels CSV (single column) |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--X-train PATH` | `X_test` | Training features CSV — used as drift reference |
| `--sensitive PATH` | `None` | Sensitive feature CSV for fairness evaluation |
| `--output PATH` | `mlens_report.html` | Output report path (`.html` or `.pdf`) |
| `--no-shap` | off | Skip SHAP explainability |
| `--no-fairness` | off | Skip fairness evaluation |
| `--no-drift` | off | Skip drift detection |

**Examples:**

```bash
# Basic audit
mlens audit model.pkl X_test.csv y_test.csv

# With fairness evaluation
mlens audit model.pkl X_test.csv y_test.csv --sensitive gender.csv

# With training data for drift reference
mlens audit model.pkl X_test.csv y_test.csv --X-train X_train.csv

# Skip drift detection, custom output
mlens audit model.pkl X_test.csv y_test.csv --no-drift --output report.html
```

### `mlens version`

Show the installed MLens version.

```bash
mlens version
```

## CSV Format

All CSV inputs follow standard format — first row is the header:

**X_test.csv**
```csv
age,income,hours_per_week
34,52000,40
29,41000,38
```

**y_test.csv** (single column, any header name)
```csv
label
1
0
```

**sensitive.csv** (single column, any header name)
```csv
gender
Male
Female
```

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Audit completed successfully |
| `1` | File not found, or model/data loading error |
