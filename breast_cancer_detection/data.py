"""Loading, validation, and exploratory summary for the breast cancer dataset."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
    "target",
]


def validate_data(df: pd.DataFrame) -> None:
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols or extra_cols:
        raise ValueError(f"Column mismatch. Missing: {missing_cols}, Unexpected: {extra_cols}")

    if df[EXPECTED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")

    if df.duplicated().any():
        raise ValueError(f"Dataset contains {df.duplicated().sum()} duplicate rows")

    feature_cols = [c for c in EXPECTED_COLUMNS if c != "target"]
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in feature_cols):
        raise ValueError("All feature columns must be numeric")

    if (df[feature_cols] < 0).any().any():
        raise ValueError("Feature columns must not contain negative values")

    if not pd.api.types.is_integer_dtype(df["target"]) or not df["target"].isin([0, 1]).all():
        raise ValueError("Target column must only contain 0 (Benign) or 1 (Malignant)")


def summarize_data(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c != "target"]
    class_counts = df["target"].map({0: "Benign", 1: "Malignant"}).value_counts().to_dict()

    corr = df[feature_cols].corr().abs()
    pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False)

    seen_pairs = set()
    top_correlations = []
    for (feature_a, feature_b), value in pairs.items():
        pair_key = tuple(sorted((feature_a, feature_b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        top_correlations.append({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "correlation": round(float(value), 3),
        })
        if len(top_correlations) == 5:
            break

    return {
        "class_counts": class_counts,
        "feature_stats": df[feature_cols].describe().to_dict(),
        "top_correlations": top_correlations,
    }


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_data(df)

    label_map = {0: "Benign", 1: "Malignant"}
    logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1] - 1} features")
    logger.info(f"Class distribution:\n{df['target'].map(label_map).value_counts().to_string()}\n")

    return df
