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
