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
