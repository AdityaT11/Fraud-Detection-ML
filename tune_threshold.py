"""Analyze fraud thresholds on a separate validation split."""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)
from src.preprocessing import load_dataset, split_features_target

df = load_dataset("data/transactions.csv")
X, y = split_features_target(df)

# Reproduce the training split, then reserve part of training data
# for validation. The original test set remains untouched.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2,
    random_state=43, stratify=y_train
)

# Train a fresh model on the reduced training portion.
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import build_preprocessor

model = Pipeline([
    ("preprocessing", build_preprocessor()),
    ("undersampling", RandomUnderSampler(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_fit, y_fit)
scores = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, scores)
pr_auc = average_precision_score(y_val, scores)

print(f"Validation PR-AUC: {pr_auc:.4f}")
print("\nThreshold | Precision | Recall | F1")
print("-" * 43)

rows = []
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (scores >= threshold).astype(int)
    tp = ((pred == 1) & (y_val.to_numpy() == 1)).sum()
    fp = ((pred == 1) & (y_val.to_numpy() == 0)).sum()
    fn = ((pred == 0) & (y_val.to_numpy() == 1)).sum()

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    rows.append({
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    })
    print(f"{threshold:9.2f} | {p:9.3f} | {r:6.3f} | {f1:.3f}")

pd.DataFrame(rows).to_csv("reports/threshold_analysis.csv", index=False)
print("\nSaved reports/threshold_analysis.csv")
