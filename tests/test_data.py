import numpy as np
import pandas as pd
import pytest

from breast_cancer_detection.data import EXPECTED_COLUMNS, validate_data, summarize_data


def _good_df():
    n = 5
    data = {col: np.linspace(1.0, 10.0, n) for col in EXPECTED_COLUMNS if col != "target"}
    data["target"] = [0, 1, 0, 1, 0]
    return pd.DataFrame(data)


def test_validate_data_passes_on_good_data():
    validate_data(_good_df())  # should not raise


def test_validate_data_rejects_missing_column():
    df = _good_df().drop(columns=["mean radius"])
    with pytest.raises(ValueError, match="Column mismatch"):
        validate_data(df)


def test_validate_data_rejects_missing_values():
    df = _good_df()
    df.loc[0, "mean radius"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_data(df)


def test_validate_data_rejects_duplicate_rows():
    df = _good_df()
    df.loc[len(df)] = df.loc[0]
    with pytest.raises(ValueError, match="duplicate rows"):
        validate_data(df)


def test_validate_data_rejects_negative_values():
    df = _good_df()
    df.loc[0, "mean radius"] = -1.0
    with pytest.raises(ValueError, match="negative values"):
        validate_data(df)


def test_validate_data_rejects_bad_target():
    df = _good_df()
    df.loc[0, "target"] = 2
    with pytest.raises(ValueError, match="Target column"):
        validate_data(df)


def test_summarize_data_returns_expected_keys():
    summary = summarize_data(_good_df())
    assert set(summary.keys()) == {"class_counts", "feature_stats", "top_correlations"}
    assert summary["class_counts"] == {"Benign": 3, "Malignant": 2} or summary["class_counts"] == {"Malignant": 2, "Benign": 3}
    assert len(summary["top_correlations"]) <= 5
    for pair in summary["top_correlations"]:
        assert set(pair.keys()) == {"feature_a", "feature_b", "correlation"}
