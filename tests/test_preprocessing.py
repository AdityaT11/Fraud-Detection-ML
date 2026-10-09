import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessor,
    load_dataset,
    split_features_target,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "nameOrig": ["C1", "C2", "C3", "C4"],
            "oldbalanceOrg": [500.0, 200.0, 300.0, 400.0],
            "newbalanceOrig": [400.0, 0.0, 0.0, 400.0],
            "nameDest": ["M1", "M2", "M3", "M4"],
            "oldbalanceDest": [0.0, 100.0, 200.0, 0.0],
            "newbalanceDest": [0.0, 300.0, 500.0, 0.0],
            "isFraud": [0, 1, 1, 0],
            "isFlaggedFraud": [0, 0, 0, 0],
        }
    )


def test_load_dataset_removes_identifiers(tmp_path, sample_data):
    path = tmp_path / "sample.csv"
    sample_data.to_csv(path, index=False)

    df = load_dataset(path)

    assert "nameOrig" not in df.columns
    assert "nameDest" not in df.columns
    assert "isFlaggedFraud" not in df.columns
    assert "isFraud" in df.columns


def test_preprocessor_returns_numeric_matrix(sample_data):
    df = sample_data.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    X, y = split_features_target(df)

    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(y)
    assert transformed.shape[1] > len(X.columns)
    assert set(y.unique()) == {0, 1}


def test_missing_required_columns_raise_error(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"amount": [100]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
