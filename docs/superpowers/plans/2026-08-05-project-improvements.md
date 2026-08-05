# Breast Cancer Detection: Project Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the single-file breast cancer detection script into a tested package with validated CSV-based data, F2/HalvingGridSearchCV-tuned model, and a self-contained HTML report.

**Architecture:** Split `breast_cancer_detection.py` into a `breast_cancer_detection/` package (`config`, `data`, `model`, `report`, `cli`) driven by a thin `main.py`, backed by a checked-in `data/breast_cancer_wisconsin.csv` and covered by `pytest` unit tests for pure-logic functions.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn (HalvingGridSearchCV, permutation_importance), xgboost, matplotlib, joblib, pytest. No new dependencies except `pytest`.

## Global Constraints

- `RANDOM_STATE` default is `42`, `TEST_SIZE` default is `0.2` — unchanged from the current script.
- Class encoding is `0 = Benign, 1 = Malignant` everywhere (data, metrics, report) — must not change.
- No new runtime dependencies. `pytest` is the only new entry in `requirements.txt` (test-only).
- No Jinja2 or other templating library — HTML is built with plain f-strings.
- The old `breast_cancer_detection.py` flat script is deleted once its logic has been migrated (Task 6) — a package directory and a module file with the same name cannot coexist.
- Dependencies already confirmed installable on this machine's Python 3.14: scikit-learn 1.9.0, xgboost 3.4.0, pandas 3.0.5, numpy 2.5.1, matplotlib 3.11.1, joblib 1.5.3, pytest 9.1.1 (verified via `pip install`).

---

### Task 1: Package scaffold, config, checked-in dataset CSV

**Files:**
- Create: `breast_cancer_detection/__init__.py`
- Create: `breast_cancer_detection/config.py`
- Create: `data/breast_cancer_wisconsin.csv`
- Create: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `Config` dataclass with fields `RANDOM_STATE: int`, `TEST_SIZE: float`, `DATA_PATH: Path`, `OUTPUT_DIR: Path`, and properties `REPORT_PATH: Path`, `MODEL_PATH: Path`. Later tasks import `from breast_cancer_detection.config import Config`.

- [ ] **Step 1: Create package and data directories**

```bash
mkdir -p breast_cancer_detection data tests
```

- [ ] **Step 2: Create empty package init**

Create `breast_cancer_detection/__init__.py` with no content (empty file).

- [ ] **Step 3: Write config.py**

```python
"""Centralized configuration for hyperparameters, splits, and file paths."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    DATA_PATH: Path = Path("data/breast_cancer_wisconsin.csv")
    OUTPUT_DIR: Path = Path(".")

    @property
    def REPORT_PATH(self) -> Path:
        return self.OUTPUT_DIR / "breast_cancer_report.html"

    @property
    def MODEL_PATH(self) -> Path:
        return self.OUTPUT_DIR / "breast_cancer_pipeline.joblib"
```

- [ ] **Step 4: Export the dataset to a checked-in CSV**

Run this one-off command (not saved as a script — it runs once):

```bash
python -c "
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
df = X.copy()
df['target'] = 1 - y  # 0 = Benign, 1 = Malignant, matching clinical convention
df.to_csv('data/breast_cancer_wisconsin.csv', index=False)
print('wrote', df.shape)
"
```

Expected output: `wrote (569, 31)`

- [ ] **Step 5: Verify the CSV**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/breast_cancer_wisconsin.csv')
assert df.shape == (569, 31), df.shape
assert set(df['target'].unique()) == {0, 1}
print('OK', df.shape)
"
```

Expected: `OK (569, 31)`

- [ ] **Step 6: Add .gitignore for generated outputs**

Create `.gitignore`:

```
__pycache__/
*.pyc
breast_cancer_report.html
breast_cancer_pipeline.joblib
breast_cancer_evaluation.png
.pytest_cache/
```

- [ ] **Step 7: Add pytest to requirements.txt**

Modify `requirements.txt`, add a line at the end:

```
pytest>=7.0.0
```

- [ ] **Step 8: Commit**

```bash
git add breast_cancer_detection/ data/ .gitignore requirements.txt
git commit -m "Add package scaffold, config, checked-in dataset CSV"
```

---

### Task 2: `data.py` — load, validate, summarize (TDD)

**Files:**
- Create: `breast_cancer_detection/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from other package modules (only `pandas`, `numpy`, stdlib).
- Produces: `EXPECTED_COLUMNS: list[str]` (30 feature names + `"target"`), `validate_data(df: pd.DataFrame) -> None` (raises `ValueError` on any failure), `summarize_data(df: pd.DataFrame) -> dict` with keys `"class_counts"`, `"feature_stats"`, `"top_correlations"` (list of `{"feature_a", "feature_b", "correlation"}`, max 5, deduplicated), `load_data(path: Path) -> pd.DataFrame` (reads CSV, validates, logs, returns).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'breast_cancer_detection.data'` (or collection error) — the module doesn't exist yet.

- [ ] **Step 3: Write `data.py`**

```python
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

    if not df["target"].isin([0, 1]).all():
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_data.py -v
```

Expected: all tests `PASS`

- [ ] **Step 5: Commit**

```bash
git add breast_cancer_detection/data.py tests/test_data.py
git commit -m "Add data.py: CSV loading, schema validation, EDA summary"
```

---

### Task 3: `model.py` — testable helpers (F2 scorer, class imbalance ratio) (TDD)

**Files:**
- Create: `breast_cancer_detection/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `compute_class_imbalance_ratio(y: pd.Series) -> float`, `make_f2_scorer() -> sklearn scorer callable` (usable as `scoring=` in a search). Both live in `breast_cancer_detection/model.py`, which Task 4 extends with `build_and_tune_model` and `evaluate_model`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

from breast_cancer_detection.model import compute_class_imbalance_ratio, make_f2_scorer


def test_compute_class_imbalance_ratio():
    y = pd.Series([0, 0, 0, 0, 1])
    assert compute_class_imbalance_ratio(y) == pytest.approx(4.0)


def test_make_f2_scorer_matches_fbeta_score():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    scorer = make_f2_scorer()

    class DummyModel:
        def predict(self, X):
            return np.array(y_pred)

    score = scorer(DummyModel(), None, y_true)
    expected = fbeta_score(y_true, y_pred, beta=2)
    assert score == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'breast_cancer_detection.model'`

- [ ] **Step 3: Write `model.py` with the two helpers**

```python
"""Model pipeline: hyperparameter tuning, evaluation, and their supporting metrics."""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer

logger = logging.getLogger(__name__)


def compute_class_imbalance_ratio(y: pd.Series) -> float:
    neg_count, pos_count = np.bincount(y)
    return neg_count / pos_count


def make_f2_scorer():
    return make_scorer(fbeta_score, beta=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model.py -v
```

Expected: both tests `PASS`

- [ ] **Step 5: Commit**

```bash
git add breast_cancer_detection/model.py tests/test_model.py
git commit -m "Add model.py helpers: F2 scorer, class imbalance ratio"
```

---

### Task 4: `model.py` — training and evaluation (HalvingGridSearchCV, permutation importance)

**Files:**
- Modify: `breast_cancer_detection/model.py` (append to the file created in Task 3)

**Interfaces:**
- Consumes: `compute_class_imbalance_ratio`, `make_f2_scorer` (Task 3, same file); `Config` fields `RANDOM_STATE` (Task 1) passed in by the caller, not imported directly.
- Produces: `build_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> tuple[Pipeline, HalvingGridSearchCV]`; `evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, random_state: int) -> tuple[np.ndarray, np.ndarray, dict, pd.Series]` where the returned tuple is `(y_pred, y_proba, metrics, feature_importance)` and `feature_importance` is a `pd.Series` indexed by feature name, sorted descending. `metrics` keys: `"Accuracy"`, `"Precision"`, `"Recall"`, `"F1-Score"`, `"F2-Score"`, `"ROC-AUC"`.

This task has no unit tests — `HalvingGridSearchCV` training is slow and its correctness is sklearn's responsibility, not ours. It is verified manually against the real dataset in Step 2.

- [ ] **Step 1: Append training and evaluation functions to `model.py`**

Add these imports to the top of `breast_cancer_detection/model.py` (alongside the existing ones from Task 3):

```python
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (registers HalvingGridSearchCV)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
```

Append these functions to the end of the file:

```python
def build_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    base_ratio = compute_class_imbalance_ratio(y_train)
    logger.info(f"Class imbalance ratio (benign : malignant) in training set: {base_ratio:.2f} : 1\n")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "xgb__max_depth": [3, 4, 5],
        "xgb__learning_rate": [0.01, 0.1, 0.2],
        "xgb__n_estimators": [100, 200],
        "xgb__scale_pos_weight": [1, round(base_ratio, 2), round(base_ratio * 1.5, 2)],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_f2_scorer(),
        refit=True,
        cv=cv_strategy,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Starting HalvingGridSearchCV hyperparameter tuning (F2-scored)...")
    search.fit(X_train, y_train)

    logger.info(f"Best hyperparameters found: {search.best_params_}")
    logger.info(f"Best cross-validated F2 score: {search.best_score_:.4f}\n")
    logger.info(f"Candidates evaluated: {len(search.cv_results_['params'])}\n")

    return search.best_estimator_, search


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, random_state: int):
    y_pred = np.asarray(model.predict(X_test))
    y_proba = np.asarray(model.predict_proba(X_test))[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1-Score": float(f1_score(y_test, y_pred)),
        "F2-Score": float(fbeta_score(y_test, y_pred, beta=2)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }
    logger.info(f"Test Set Metrics:\n{json.dumps(metrics, indent=4)}")

    importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    feature_importance = pd.Series(
        importance.importances_mean, index=X_test.columns
    ).sort_values(ascending=False)

    return y_pred, y_proba, metrics, feature_importance
```

- [ ] **Step 2: Manually verify end-to-end against the real dataset**

```bash
python -c "
from pathlib import Path
from sklearn.model_selection import train_test_split

from breast_cancer_detection.data import load_data
from breast_cancer_detection.model import build_and_tune_model, evaluate_model

df = load_data(Path('data/breast_cancer_wisconsin.csv'))
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model, search = build_and_tune_model(X_train, y_train, random_state=42)
y_pred, y_proba, metrics, importance = evaluate_model(model, X_test, y_test, random_state=42)

print('Metrics:', metrics)
print('Top feature:', importance.index[0])
assert all(0.0 <= v <= 1.0 for v in metrics.values())
print('OK')
"
```

Expected: prints a `Metrics` dict with all six values between 0 and 1, a `Top feature` name, and `OK`. No exceptions.

- [ ] **Step 3: Commit**

```bash
git add breast_cancer_detection/model.py
git commit -m "Add HalvingGridSearchCV tuning and evaluation with permutation importance"
```

---

### Task 5: `report.py` — HTML report generation

**Files:**
- Create: `breast_cancer_detection/report.py`

**Interfaces:**
- Consumes: nothing from other package modules directly — receives all data as function arguments (matplotlib, pandas, numpy, sklearn.metrics only).
- Produces: `generate_report(df, X_test, y_test, y_pred, y_proba, metrics, feature_importance, data_summary, best_params, candidates_evaluated, output_path) -> Path`. Argument types match what Task 4 (`metrics`, `feature_importance`) and Task 2 (`data_summary` from `summarize_data`) produce, plus `search.best_params_` (dict) and `len(search.cv_results_['params'])` (int) from Task 4's `search` object.

No unit tests — this is charting/HTML generation, verified visually. Verification is manual in Step 2.

- [ ] **Step 1: Write `report.py`**

```python
"""HTML report generation: matplotlib charts embedded as base64 images plus metrics tables."""
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _plot_class_balance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df["target"].map({0: "Benign", 1: "Malignant"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Class Balance")
    ax.set_ylabel("Sample Count")
    return _fig_to_base64(fig)


def _plot_confusion_matrix(y_test, y_pred) -> str:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Benign", "Malignant"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_base64(fig)


def _plot_precision_recall(y_test, y_proba) -> str:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision, color="#e76f51", lw=2, label=f"AP = {ap_score:.3f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#e76f51")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def _plot_roc_curve(y_test, y_proba) -> str:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#264653", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def _plot_feature_importance(feature_importance: pd.Series, top_n: int = 10) -> str:
    top = feature_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.barh(top.index, top.values, color="#2a9d8f")
    ax.set_title(f"Top {top_n} Feature Importances (Permutation)")
    ax.set_xlabel("Mean Importance Decrease")
    return _fig_to_base64(fig)


def _metrics_table_html(metrics: Dict[str, float]) -> str:
    tiles = "".join(
        f'<div class="tile"><div class="tile-value">{value:.3f}</div>'
        f'<div class="tile-label">{name}</div></div>'
        for name, value in metrics.items()
    )
    return f'<div class="tiles">{tiles}</div>'


def _correlations_table_html(top_correlations: List[dict]) -> str:
    rows = "".join(
        f'<tr><td>{c["feature_a"]}</td><td>{c["feature_b"]}</td><td>{c["correlation"]:.3f}</td></tr>'
        for c in top_correlations
    )
    return f"""
    <table>
        <thead><tr><th>Feature A</th><th>Feature B</th><th>|Correlation|</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


_STYLE = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: #f7f7f5; color: #1f2937; margin: 0; padding: 2rem; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
.meta { color: #6b7280; margin-bottom: 2rem; }
h2 { font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem; }
.tiles { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }
.tile { background: white; border-radius: 10px; padding: 1rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); min-width: 120px; text-align: center; }
.tile-value { font-size: 1.6rem; font-weight: 700; color: #264653; }
.tile-label { color: #6b7280; font-size: 0.85rem; margin-top: 0.25rem; }
.charts { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 1rem; }
.charts img { background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 0.5rem; max-width: 100%; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; background: white; border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 0.6rem 1rem; border-bottom: 1px solid #e5e7eb; }
th { background: #264653; color: white; }
.params { background: white; border-radius: 10px; padding: 1rem 1.5rem; margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
"""


def generate_report(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics: Dict[str, float],
    feature_importance: pd.Series,
    data_summary: dict,
    best_params: dict,
    candidates_evaluated: int,
    output_path: Path,
) -> Path:
    class_balance_img = _plot_class_balance(df)
    cm_img = _plot_confusion_matrix(y_test, y_pred)
    pr_img = _plot_precision_recall(y_test, y_proba)
    roc_img = _plot_roc_curve(y_test, y_proba)
    importance_img = _plot_feature_importance(feature_importance)

    params_html = "".join(f"<div><strong>{k}</strong>: {v}</div>" for k, v in best_params.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Breast Cancer Detection Report</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="container">
    <h1>Breast Cancer Detection Report</h1>
    <div class="meta">
        Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &middot;
        {df.shape[0]} total samples &middot; {X_test.shape[0]} test samples
    </div>

    <h2>Model Performance</h2>
    {_metrics_table_html(metrics)}

    <h2>Dataset Overview</h2>
    <div class="charts"><img src="data:image/png;base64,{class_balance_img}" alt="Class balance"></div>
    <h3>Most Correlated Feature Pairs</h3>
    {_correlations_table_html(data_summary["top_correlations"])}

    <h2>Evaluation Charts</h2>
    <div class="charts">
        <img src="data:image/png;base64,{cm_img}" alt="Confusion matrix">
        <img src="data:image/png;base64,{pr_img}" alt="Precision-recall curve">
        <img src="data:image/png;base64,{roc_img}" alt="ROC curve">
    </div>

    <h2>Feature Importance</h2>
    <div class="charts"><img src="data:image/png;base64,{importance_img}" alt="Feature importance"></div>

    <h2>Best Hyperparameters</h2>
    <div class="params">
        {params_html}
        <div style="margin-top: 0.5rem; color: #6b7280;">{candidates_evaluated} candidates evaluated</div>
    </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
```

- [ ] **Step 2: Manually verify report generation**

```bash
python -c "
from pathlib import Path
from sklearn.model_selection import train_test_split

from breast_cancer_detection.data import load_data, summarize_data
from breast_cancer_detection.model import build_and_tune_model, evaluate_model
from breast_cancer_detection.report import generate_report

df = load_data(Path('data/breast_cancer_wisconsin.csv'))
data_summary = summarize_data(df)
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model, search = build_and_tune_model(X_train, y_train, random_state=42)
y_pred, y_proba, metrics, importance = evaluate_model(model, X_test, y_test, random_state=42)

path = generate_report(
    df=df, X_test=X_test, y_test=y_test, y_pred=y_pred, y_proba=y_proba,
    metrics=metrics, feature_importance=importance, data_summary=data_summary,
    best_params=search.best_params_, candidates_evaluated=len(search.cv_results_['params']),
    output_path=Path('breast_cancer_report.html'),
)
content = path.read_text(encoding='utf-8')
assert 'Breast Cancer Detection Report' in content
assert 'Confusion Matrix' in content
assert path.stat().st_size > 10000
print('OK', path, path.stat().st_size, 'bytes')
"
```

Expected: `OK breast_cancer_report.html <size> bytes` with no exceptions. Open `breast_cancer_report.html` in a browser and confirm it renders the metrics tiles, 5 charts, correlation table, and hyperparameters cleanly.

- [ ] **Step 3: Commit**

```bash
git add breast_cancer_detection/report.py
git commit -m "Add HTML report generation with embedded charts"
```

---

### Task 6: `cli.py` + `main.py` — wire the pipeline together, remove the old script

**Files:**
- Create: `breast_cancer_detection/cli.py`
- Create: `main.py`
- Delete: `breast_cancer_detection.py` (the old flat script — its logic is now fully covered by Tasks 2–5)

**Interfaces:**
- Consumes: `Config` (Task 1), `load_data`/`summarize_data` (Task 2), `build_and_tune_model`/`evaluate_model` (Task 4), `generate_report` (Task 5).
- Produces: `parse_args() -> Config`, `main() -> None` in `cli.py`. `main.py` is a two-line entrypoint.

- [ ] **Step 1: Delete the old flat script**

```bash
rm breast_cancer_detection.py
```

- [ ] **Step 2: Write `cli.py`**

```python
"""Command-line entrypoint: argument parsing and full pipeline orchestration."""
import argparse
import logging
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .config import Config
from .data import load_data, summarize_data
from .model import build_and_tune_model, evaluate_model
from .report import generate_report


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train and evaluate the breast cancer detection model.")
    parser.add_argument("--test-size", type=float, default=Config.TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=Config.RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=Config.OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=Config.DATA_PATH)
    args = parser.parse_args()

    return Config(
        RANDOM_STATE=args.random_state,
        TEST_SIZE=args.test_size,
        DATA_PATH=args.data_path,
        OUTPUT_DIR=args.output_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    cfg = parse_args()

    df = load_data(cfg.DATA_PATH)
    data_summary = summarize_data(df)

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
    )
    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

    best_pipeline, search = build_and_tune_model(X_train, y_train, random_state=cfg.RANDOM_STATE)
    y_pred, y_proba, metrics, feature_importance = evaluate_model(
        best_pipeline, X_test, y_test, random_state=cfg.RANDOM_STATE
    )

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = generate_report(
        df=df,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        feature_importance=feature_importance,
        data_summary=data_summary,
        best_params=search.best_params_,
        candidates_evaluated=len(search.cv_results_["params"]),
        output_path=cfg.REPORT_PATH,
    )
    logger.info(f"HTML report saved to: {report_path}")

    joblib.dump(best_pipeline, cfg.MODEL_PATH)
    logger.info(f"Complete inference pipeline saved to: {cfg.MODEL_PATH}")
```

- [ ] **Step 3: Write `main.py`**

```python
from breast_cancer_detection.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full pipeline end-to-end**

```bash
python main.py
```

Expected: runs to completion with no exceptions, logs train/test split sizes, tuning progress, best hyperparameters, and metrics; ends with `HTML report saved to: breast_cancer_report.html` and `Complete inference pipeline saved to: breast_cancer_pipeline.joblib`.

- [ ] **Step 5: Verify the generated artifacts**

```bash
python -c "
from pathlib import Path
assert Path('breast_cancer_report.html').exists()
assert Path('breast_cancer_pipeline.joblib').exists()
print('OK')
"
```

Expected: `OK`

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest -v
```

Expected: all tests in `tests/test_data.py` and `tests/test_model.py` `PASS`

- [ ] **Step 7: Commit**

```bash
git add -A -- breast_cancer_detection/cli.py main.py
git rm breast_cancer_detection.py
git commit -m "Wire CLI/pipeline together, remove old flat script"
```

---

### Task 7: README rewrite

**Files:**
- Modify: `README.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Rewrite `README.md`**

Replace the full content of `README.md` with:

```markdown
# Breast Cancer Detection Pipeline

An end-to-end machine learning pipeline that classifies breast tumors as Benign or Malignant.

This project prioritizes **Recall** (via an F2-weighted tuning objective) to minimize false negatives, uses `scikit-learn` Pipelines to prevent data leakage, and tunes hyperparameters with `HalvingGridSearchCV`. Every run produces a single self-contained HTML report with metrics, EDA, and evaluation charts.

## Project Objective & Clinical Context

In medical diagnostics, failing to detect a malignant tumor (a False Negative) carries a drastically higher cost than a false alarm (a False Positive).

To reflect this, the model is tuned against an **F2 score** (recall weighted above precision) rather than pure recall — pure recall can be trivially maximized by a model that predicts "Malignant" for every sample, which F2 penalizes through its precision component. Class imbalance is additionally handled via a dynamically computed `scale_pos_weight`.

## Tech Stack & Architecture

- **Language:** Python 3.9+
- **Libraries:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `pytest`
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset, checked into the repo at `data/breast_cancer_wisconsin.csv`
- **Model:** XGBoost Classifier (`XGBClassifier`)

### Project Structure

```
breast_cancer_detection/
├── config.py   # Run configuration (paths, split size, random state)
├── data.py     # CSV loading, schema validation, EDA summary
├── model.py    # HalvingGridSearchCV tuning, evaluation, permutation importance
├── report.py   # Self-contained HTML report generation
├── cli.py      # Argument parsing and pipeline orchestration
data/
└── breast_cancer_wisconsin.csv
tests/
├── test_data.py
└── test_model.py
main.py         # Entrypoint
```

### Pipeline Features

1. **Validated input:** the dataset is loaded from a checked-in CSV and validated on every run (schema, missing values, duplicates, value ranges) before training.
2. **EDA on every run:** class balance and top correlated feature pairs are computed and surfaced in the HTML report.
3. **Leak-proof preprocessing:** `StandardScaler` is bound inside a `scikit-learn` `Pipeline`, fit only on training folds during cross-validation.
4. **Efficient tuning:** `HalvingGridSearchCV` prunes weak hyperparameter candidates early instead of exhaustively evaluating every combination on the full training set.
5. **F2-scored objective + dynamic class balancing:** tuning optimizes F2 (recall-weighted) and searches over `scale_pos_weight` values derived from the training set's actual class ratio.
6. **Self-contained HTML report:** metrics, EDA charts, confusion matrix, PR curve, ROC curve, feature importance, and the winning hyperparameters — one file, no external images.

## How to Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/Vandahlen/breast-cancer-detection.git
cd breast-cancer-detection
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the pipeline:**

```bash
python main.py
```

Optional flags: `--test-size`, `--random-state`, `--output-dir`, `--data-path`. Example:

```bash
python main.py --test-size 0.25 --random-state 7 --output-dir results/
```

This produces `breast_cancer_report.html` (open it in a browser) and `breast_cancer_pipeline.joblib` (the trained inference pipeline) in the chosen output directory.

**4. Run the tests:**

```bash
python -m pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Rewrite README for new package structure, tuning, and HTML report"
```

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 1), Data Handling (Task 2), Model Efficiency (Tasks 3–4), HTML Report (Task 5), CLI/Config (Task 6), Testing (Tasks 2–3, verified in Task 6 Step 6), README (Task 7) — all spec sections have a task.
- **Type consistency checked:** `evaluate_model` returns `(y_pred, y_proba, metrics, feature_importance)` consistently in Tasks 4, 5's verification script, and Task 6's `cli.py`. `generate_report`'s parameter names match what Task 6 passes (`data_summary`, `best_params`, `candidates_evaluated`, `output_path`). `Config.REPORT_PATH`/`MODEL_PATH` (Task 1) match what `cli.py` (Task 6) reads.
- **No placeholders:** every step has complete, runnable code or an exact command.
