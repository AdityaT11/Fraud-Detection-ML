"""Train and evaluate fraud-detection models."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    TARGET,
    build_preprocessor,
    load_dataset,
    split_features_target,
)

MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports")


def evaluate_model(model, X_test, y_test, name):
    """Evaluate a fitted model and save its metrics and confusion matrix."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": name,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print(f"\n=== {name} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Fraud", "Fraud"],
        zero_division=0,
    ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Not Fraud", "Fraud"],
        values_format="d",
    )
    display.ax_.set_title(f"Confusion Matrix — {name}")
    display.figure_.tight_layout()
    display.figure_.savefig(RESULTS_DIR / f"{name.lower().replace(' ', '_')}_confusion.png")
    plt.close(display.figure_)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/transactions.csv",
        help="Path to the full transaction CSV",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    df = load_dataset(args.data)
    X, y = split_features_target(df)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("The dataset needs at least two examples of each class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"Dataset rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print("Original training class counts:")
    print(y_train.value_counts().sort_index().to_dict())

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=args.random_state,
            n_jobs=-1,
        ),
    }

    all_metrics = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, estimator in models.items():
        pipeline = ImbPipeline(
            steps=[
                ("preprocessing", build_preprocessor()),
                (
                    "undersampling",
                    RandomUnderSampler(random_state=args.random_state),
                ),
                ("classifier", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        all_metrics.append(metrics)

        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / model_filename)
        print(f"Saved model: {MODEL_DIR / model_filename}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(all_metrics).sort_values("pr_auc", ascending=False)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\n=== Model comparison (sorted by PR-AUC) ===")
    print(results.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/model_comparison.csv")


if __name__ == "__main__":
    main()
