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


def test_validate_data_rejects_non_numeric_feature():
    df = _good_df()
    df["mean radius"] = df["mean radius"].astype(object)
    df.loc[0, "mean radius"] = "not-a-number"
    with pytest.raises(ValueError, match="must be numeric"):
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


def test_validate_data_rejects_float_target():
    df = _good_df()
    df["target"] = df["target"].astype(float)
    with pytest.raises(ValueError, match="Target column"):
        validate_data(df)


def _varied_df():
    n = 6
    data = {
        col: [(i * (idx + 1)) % 7 + i * 0.37 for i in range(n)]
        for idx, col in enumerate(c for c in EXPECTED_COLUMNS if c != "target")
    }
    data["target"] = [0, 1, 0, 1, 0, 1]
    return pd.DataFrame(data)


def test_summarize_data_returns_expected_keys():
    summary = summarize_data(_varied_df())
    assert set(summary.keys()) == {"class_counts", "feature_stats", "top_correlations"}
    assert dict(summary["class_counts"]) == {"Benign": 3, "Malignant": 3}
    assert 0 < len(summary["top_correlations"]) <= 5
    seen_pairs = set()
    for pair in summary["top_correlations"]:
        assert set(pair.keys()) == {"feature_a", "feature_b", "correlation"}
        pair_key = tuple(sorted((pair["feature_a"], pair["feature_b"])))
        assert pair_key not in seen_pairs, "top_correlations must not contain a mirrored duplicate pair"
        seen_pairs.add(pair_key)
