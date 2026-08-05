# Breast Cancer Detection: Project Improvements Design

Date: 2026-08-05

## Goals

- Improve model efficiency: both predictive quality (avoid the pure-recall degenerate solution) and tuning speed.
- Move off sklearn's hidden bundled dataset onto a checked-in, validated CSV with an EDA step.
- Replace the two-panel PNG output with a single, visually polished, self-contained HTML report.
- Restructure the single-file script into a small tested package with CLI configuration, and bring the README up to date.

## Non-Goals

- Swapping to a different/larger dataset. The Wisconsin dataset is the standard benchmark for this problem; the weakness is pipeline transparency and testing, not data volume.
- Exposing the hyperparameter grid or CV fold count as CLI flags — those are tuning internals, not run-to-run knobs.
- Full training-run integration tests (slow, low value) — testing targets pure logic only.
- Adding Jinja2 or any HTML templating library — an f-string template is sufficient for one report.

## Architecture

```
breast_cancer_detection/
├── __init__.py
├── config.py        # Config dataclass, populated from CLI args
├── data.py          # load_data, validate_data, summarize_data (EDA)
├── model.py          # build_and_tune_model, evaluate_model
├── report.py         # HTML report generation (charts + tables)
├── cli.py             # argparse entrypoint
data/
└── breast_cancer_wisconsin.csv   # checked-in copy of the dataset (0=Benign, 1=Malignant)
tests/
├── test_data.py
├── test_model.py
main.py               # thin entrypoint calling cli.main()
```

The CSV is produced once via a one-time export from `sklearn.datasets.load_breast_cancer`, preserving the existing 0=Benign/1=Malignant label convention.

## Data Handling

**Loading:** `data.py` reads `data/breast_cancer_wisconsin.csv` with pandas.

**Validation** — `validate_data(df)`, called immediately after load, raises on failure (this is a trust boundary since the CSV is a plain checked-in file that could be hand-edited):
- Expected column set and dtypes match the known schema
- No missing values
- No duplicate rows
- Each feature falls within a sane numeric range

**EDA summary** — `summarize_data(df)`, computed once after validation and reused by both the console log and the HTML report:
- Class balance (Benign vs Malignant counts)
- Per-feature summary stats (mean/std/min/max)
- Top correlated feature pairs (informational only — does not affect the pipeline)

## Model Efficiency

**Scoring:** Replace `scoring="recall"` with an F2-scorer (`fbeta_score(beta=2)` via `make_scorer`). Weights recall above precision — matching clinical priority — without allowing the "predict everyone malignant" degenerate optimum that pure recall permits.

**Search:** Replace `GridSearchCV` with `HalvingGridSearchCV` (`sklearn.experimental.enable_halving_search_cv`), same param grid and `StratifiedKFold`, same `random_state`. Successive halving eliminates weak candidates on small resource budgets before spending full-data fits on survivors — fewer total fits over the same search space.

**Evaluation:** `evaluate_model` keeps its existing metrics (Accuracy, Precision, Recall, F1, ROC-AUC), adds F2 (the tuning objective), and additionally computes permutation feature importance on the test set for the report.

## HTML Report

`report.py`: `generate_report(df, X_test, y_test, y_pred, y_proba, metrics, feature_importance, search_results) -> Path`, replacing `plot_results` in `main()`.

Single self-contained file (`breast_cancer_report.html`), containing:
- Header: run timestamp, dataset size, train/test split sizes
- Metrics table: Accuracy, Precision, Recall, F1, ROC-AUC, F2
- EDA snapshot: class balance chart, top correlated feature pairs
- Confusion Matrix + Precision-Recall Curve (restyled)
- ROC Curve (new)
- Feature Importance (new, permutation importance)
- Best hyperparameters and candidate count from the search

Charts are rendered with matplotlib, saved to `BytesIO`, base64-encoded, and inlined as `<img>` tags — no separate image files, no new dependencies. HTML/CSS is built with an f-string template using one light-theme stylesheet.

The old `breast_cancer_evaluation.png` output is removed. `joblib` model saving is unchanged.

## CLI / Config

`Config` is populated from `argparse` in `cli.py`:

```bash
python main.py --test-size 0.25 --random-state 7 --output-dir results/ --data-path data/breast_cancer_wisconsin.csv
```

Flags: `--test-size`, `--random-state`, `--output-dir` (controls both HTML report and joblib paths), `--data-path`. Param grid and CV folds remain hardcoded.

## Testing

Pytest covers pure logic only:
- `test_data.py`: `validate_data` catches each failure mode (missing column, wrong dtype, out-of-range value, duplicate row) and passes on good data; `summarize_data` produces expected keys/shape on a small fixture DataFrame.
- `test_model.py`: F2-scorer computes the expected value on a hand-built confusion matrix; class-imbalance ratio calculation is correct on a known label array.

No test trains the full model or runs `HalvingGridSearchCV` — that's covered by manual/experiment-runner runs, not unit tests.

## README

Rewritten to reflect the new package structure, CSV-based data with validation/EDA, HalvingGridSearchCV + F2 tuning, the HTML report as primary output, and the new CLI flags. Also fixes the existing file, which currently cuts off mid-sentence in "How to Run Locally".
