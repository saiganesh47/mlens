"""
examples/quickstart.py
========================
Full end-to-end demo of MLens using the UCI Adult Income dataset.

Run:
    python examples/quickstart.py
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mlens import ModelAuditor


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading Adult Income dataset …")
    dataset = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame.dropna()

    target_col     = "class"
    sensitive_col  = "sex"

    # Encode categoricals
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    cat_cols = df.select_dtypes("category").columns.tolist()
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col, sensitive_col])
    y = df[target_col]
    sensitive = df[sensitive_col]

    # ------------------------------------------------------------------
    # 2. Train / test split  (retain original sensitive column alignment)
    # ------------------------------------------------------------------
    (X_train, X_test,
     y_train, y_test,
     s_train, s_test) = train_test_split(
        X, y, sensitive,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # 3. Train a GradientBoosting model
    # ------------------------------------------------------------------
    print("Training GradientBoostingClassifier …")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # ------------------------------------------------------------------
    # 4. Run the MLens audit
    # ------------------------------------------------------------------
    auditor = ModelAuditor(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        sensitive_features=s_test,
        feature_names=list(X.columns),
        model_name="GBT-AdultIncome",
        shap_background_samples=100,
    )

    report = auditor.run()

    # ------------------------------------------------------------------
    # 5. Print summary and save report
    # ------------------------------------------------------------------
    print("\n── Audit Summary ──────────────────────────────")
    for line in report.summary_lines:
        print(" ", line)

    print("\n── Top 5 Features (SHAP) ──────────────────────")
    if report.shap_result:
        for feat in report.shap_result.top_features(n=5):
            print(f"  {feat['rank']:>2}. {feat['name']:<25} {feat['mean_abs_shap']:.4f}")

    print("\n── Fairness Flags ─────────────────────────────")
    if report.fairness_result:
        flags = report.fairness_result.flags
        if flags:
            for flag in flags:
                print(f"  ⚠  {flag}")
        else:
            print("  ✅  No fairness violations detected.")

    print("\n── Drift Status ───────────────────────────────")
    if report.drift_result:
        print(f"  Overall: {report.drift_result.overall_status}")
        drifted = report.drift_result.drifted_features()
        if drifted:
            print(f"  Drifted features ({len(drifted)}): {', '.join(drifted[:5])}")

    # Save HTML report
    report.save("mlens_audit_report.html")


if __name__ == "__main__":
    main()
