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
