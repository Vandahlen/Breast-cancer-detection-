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
