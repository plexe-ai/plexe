"""
Unit tests for workflow helper functions.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from plexe.helpers import compute_metric


# ============================================
# Metric Computation Tests
# ============================================


def test_compute_metric_accuracy():
    """Test accuracy computation."""
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 0]

    result = compute_metric(y_true, y_pred, "accuracy")
    expected = accuracy_score(y_true, y_pred)

    assert result == pytest.approx(expected)


def test_compute_metric_rmse():
    """Test RMSE computation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])

    result = compute_metric(y_true, y_pred, "rmse")
    expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    assert result == pytest.approx(expected, abs=0.01)


def test_compute_metric_f1_score():
    """Test F1 score computation."""
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]

    result = compute_metric(y_true, y_pred, "f1_score")

    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_compute_metric_unknown_raises():
    """Test unknown metric raises ValueError."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_metric(y_true, y_pred, "unknown_metric")
