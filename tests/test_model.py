import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.metrics import fbeta_score

from breast_cancer_detection.model import compute_class_imbalance_ratio, make_f2_scorer


def test_compute_class_imbalance_ratio():
    y = pd.Series([0, 0, 0, 0, 1])
    assert compute_class_imbalance_ratio(y) == pytest.approx(4.0)


def test_make_f2_scorer_matches_fbeta_score():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    scorer = make_f2_scorer()

    class DummyModel(BaseEstimator):
        def predict(self, X):
            return np.array(y_pred)

    score = scorer(DummyModel(), None, y_true)
    expected = fbeta_score(y_true, y_pred, beta=2)
    assert score == pytest.approx(expected)
