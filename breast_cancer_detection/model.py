"""Model pipeline: hyperparameter tuning, evaluation, and their supporting metrics."""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (registers HalvingGridSearchCV)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def compute_class_imbalance_ratio(y: pd.Series) -> float:
    counts = np.bincount(y)
    if len(counts) < 2 or counts[1] == 0:
        raise ValueError(f"Expected both classes (0 and 1) present in y, got counts: {counts}")
    return counts[0] / counts[1]


def make_f2_scorer():
    return make_scorer(fbeta_score, beta=2)


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
        min_resources=45,
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
        model, X_test, y_test, scoring=make_f2_scorer(), n_repeats=10, random_state=random_state, n_jobs=-1
    )
    feature_importance = pd.Series(
        importance.importances_mean, index=X_test.columns
    ).sort_values(ascending=False)

    return y_pred, y_proba, metrics, feature_importance
