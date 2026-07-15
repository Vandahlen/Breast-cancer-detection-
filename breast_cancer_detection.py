import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@dataclass
class Config:
    """Centralized configuration for hyperparameters, splits, and file paths."""
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    PLOT_PATH: Path = Path("breast_cancer_evaluation.png")
    MODEL_PATH: Path = Path("breast_cancer_pipeline.joblib")


CONFIG = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """
    Loads the UCI Breast Cancer dataset and returns a clean DataFrame.
    Reverses sklearn's native encoding to align with clinical conventions (0 = Benign, 1 = Malignant).
    """
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    df = pd.DataFrame(X).copy()
    y_series = pd.Series(y)

    df["target"] = 1 - y_series
    label_map = {0: "Benign", 1: "Malignant"}

    logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1] - 1} features")
    logger.info(f"Class distribution:\n{df['target'].map(label_map).value_counts().to_string()}\n")

    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into stratified train and test sets. 
    Scaling is handled dynamically within the Pipeline to prevent data leakage.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG.TEST_SIZE, stratify=y, random_state=CONFIG.RANDOM_STATE
    )

    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

    return X_train, X_test, y_train, y_test


def build_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, GridSearchCV]:
    """
    Constructs a Pipeline (StandardScaler + XGBClassifier) and tunes it using 
    GridSearchCV optimized for recall.
    """
    neg_count, pos_count = np.bincount(y_train)
    base_ratio = neg_count / pos_count
    logger.info(f"Class imbalance ratio (benign : malignant) in training set: {base_ratio:.2f} : 1\n")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=CONFIG.RANDOM_STATE,
            n_jobs=-1,
        ))
    ])

    param_grid = {
        "xgb__max_depth": [3, 4, 5],
        "xgb__learning_rate": [0.01, 0.1, 0.2],
        "xgb__n_estimators": [100, 200],
        "xgb__scale_pos_weight": [1, round(base_ratio, 2), round(base_ratio * 1.5, 2)],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG.RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        refit=True,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Starting GridSearchCV hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best hyperparameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validated recall: {grid_search.best_score_:.4f}\n")

    return grid_search.best_estimator_, grid_search


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Computes performance metrics and outputs them in a standardized JSON format.
    """
    y_pred = np.asarray(model.predict(X_test))
    y_proba_full = np.asarray(model.predict_proba(X_test))
    y_proba = y_proba_full[:, 1]

    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1-Score": float(f1_score(y_test, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }

    logger.info(f"Test Set Metrics:\n{json.dumps(metrics, indent=4)}")
    
    report = str(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    logger.info(f"\nClassification Report:\n{report}")

    return y_pred, y_proba, metrics


def plot_results(y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Generates and saves the Confusion Matrix and Precision-Recall Curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    labels = ["Benign", "Malignant"]
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
                fontsize=14, fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    ax2 = axes[1]
    ax2.plot(recall, precision, color="darkorange", lw=2,
             label=f"PR curve (AP = {ap_score:.3f})")
    ax2.fill_between(recall, precision, alpha=0.15, color="darkorange")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.set_xlim([0.0, 1.02])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG.PLOT_PATH, dpi=300, bbox_inches="tight")
    logger.info(f"Evaluation plots saved to: {CONFIG.PLOT_PATH}")
    plt.show()


def main() -> Tuple[Pipeline, Dict[str, float]]:
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    best_pipeline, _ = build_and_tune_model(X_train, y_train)
    y_pred, y_proba, metrics = evaluate_model(best_pipeline, X_test, y_test)
    
    plot_results(y_test, y_pred, y_proba)

    joblib.dump(best_pipeline, CONFIG.MODEL_PATH)
    logger.info(f"Complete inference pipeline saved to: {CONFIG.MODEL_PATH}")

    return best_pipeline, metrics


if __name__ == "__main__":
    main()