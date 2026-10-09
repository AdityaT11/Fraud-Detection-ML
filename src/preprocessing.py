"""Data validation and preprocessing for transaction fraud detection."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET = "isFraud"

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]

DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV and validate the required fraud-detection columns."""
    df = pd.read_csv(path)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset contains no rows.")

    if df[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values.")

    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1, False, True}):
        raise ValueError(f"{TARGET} must contain only 0 and 1.")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[TARGET] = df[TARGET].astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps that are fitted only on training data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_features_target(df: pd.DataFrame):
    """Separate input features from the fraud label."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
