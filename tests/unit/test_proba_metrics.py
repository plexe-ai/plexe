"""
Tests for probability-based metric computation.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from plexe.helpers import compute_metric_proba


def test_compute_metric_proba_roc_auc_binary():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])

    expected = roc_auc_score(y_true, y_proba)
    result = compute_metric_proba(y_true, y_proba, "roc_auc")

    assert result == pytest.approx(expected)


def test_compute_metric_proba_log_loss():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )

    expected = log_loss(y_true, y_proba)
    result = compute_metric_proba(y_true, y_proba, "log_loss")

    assert result == pytest.approx(expected)


def test_compute_metric_proba_brier_score():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.4, 0.9])

    expected = brier_score_loss(y_true, y_proba)
    result = compute_metric_proba(y_true, y_proba, "brier_score")

    assert result == pytest.approx(expected)


def test_compute_metric_proba_brier_aliases():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.4, 0.9])

    expected = brier_score_loss(y_true, y_proba)
    assert compute_metric_proba(y_true, y_proba, "brier") == pytest.approx(expected)
    assert compute_metric_proba(y_true, y_proba, "brier_score_loss") == pytest.approx(expected)


def test_compute_metric_proba_roc_auc_ovr_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ]
    )

    expected = roc_auc_score(y_true, y_proba, multi_class="ovr")
    result = compute_metric_proba(y_true, y_proba, "roc_auc_ovr")

    assert result == pytest.approx(expected)


def test_compute_metric_proba_roc_auc_ovo_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ]
    )

    expected = roc_auc_score(y_true, y_proba, multi_class="ovo")
    result = compute_metric_proba(y_true, y_proba, "roc_auc_ovo")

    assert result == pytest.approx(expected)


def test_compute_metric_proba_brier_score_multiclass_dataframe_order():
    y_true = np.array(["b", "a", "c", "b"])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.2, 0.2],
        ]
    )
    proba_df = pd.DataFrame(y_proba, columns=["proba_b", "proba_a", "proba_c"])

    expected_one_hot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    expected = np.mean(np.sum((y_proba - expected_one_hot) ** 2, axis=1))

    result = compute_metric_proba(y_true, proba_df, "brier_score")

    assert result == pytest.approx(expected)
