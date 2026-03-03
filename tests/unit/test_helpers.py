"""
Unit tests for workflow helper functions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from plexe.config import DEFAULT_MODEL_TYPES, ModelType, detect_installed_frameworks
from plexe.helpers import (
    _evaluate_predictor,
    compute_metric,
    metric_requires_probabilities,
    normalize_probability_predictions,
    select_viable_model_types,
)
from plexe.models import DataLayout


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


def test_select_viable_model_types_defaults_image():
    """Default model types intersect with IMAGE_PATH."""
    result = select_viable_model_types(DataLayout.IMAGE_PATH)

    installed = detect_installed_frameworks()
    expected = [mt for mt in DEFAULT_MODEL_TYPES if mt in {ModelType.KERAS, ModelType.PYTORCH}]
    if not expected:
        pytest.skip(f"No image-capable frameworks installed. Installed: {installed}")

    assert result == expected


def test_select_viable_model_types_no_intersection():
    """No compatible frameworks should raise ValueError."""
    with pytest.raises(ValueError, match="No compatible model types"):
        select_viable_model_types(DataLayout.TEXT_STRING, selected_frameworks=[ModelType.XGBOOST])


def test_compute_metric_map_grouped():
    """MAP should compute per-group and average."""
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7])
    group_ids = np.array([1, 1, 2, 2])

    result = compute_metric(y_true, y_pred, "map", group_ids=group_ids)

    assert result == pytest.approx(0.75)


def test_metric_requires_probabilities():
    assert metric_requires_probabilities("roc_auc")
    assert metric_requires_probabilities("log_loss")
    assert not metric_requires_probabilities("accuracy")


def test_normalize_probability_predictions_binary_matrix_to_positive_scores():
    y_true = np.array([0, 1, 1, 0])
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])

    normalized = normalize_probability_predictions(y_true, probs, "roc_auc")

    assert normalized.shape == (4,)
    assert np.allclose(normalized, np.array([0.2, 0.9, 0.8, 0.1]))


def test_normalize_probability_predictions_multiclass_keeps_matrix():
    y_true = np.array([0, 1, 2])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    normalized = normalize_probability_predictions(y_true, probs, "roc_auc_ovr")

    assert normalized.shape == (3, 3)
    assert np.allclose(normalized, probs)


def test_normalize_probability_predictions_multiclass_raises_on_1d():
    y_true = np.array([0, 1, 2])
    probs = np.array([0.2, 0.6, 0.4])

    with pytest.raises(ValueError, match="per-class probabilities"):
        normalize_probability_predictions(y_true, probs, "log_loss")


def test_normalize_probability_predictions_multiclass_raises_on_extra_columns():
    y_true = np.array([0, 1, 2])
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ]
    )

    with pytest.raises(ValueError, match="Probability outputs have 4 columns"):
        normalize_probability_predictions(y_true, probs, "roc_auc_ovr")


def test_normalize_probability_predictions_multiclass_raises_on_single_column_matrix():
    y_true = np.array([0, 1, 2])
    probs = np.array([[0.2], [0.5], [0.3]])

    with pytest.raises(ValueError, match="Probability outputs have 1 column"):
        normalize_probability_predictions(y_true, probs, "log_loss")


def test_normalize_probability_predictions_raises_when_validation_missing_class():
    y_true = np.array([0, 1, 1, 0])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.6, 0.2],
            [0.7, 0.2, 0.1],
        ]
    )

    with pytest.raises(ValueError, match="validation labels contain 2 distinct classes"):
        normalize_probability_predictions(y_true, probs, "log_loss")


class _DummySparkDataFrame:
    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf

    def toPandas(self) -> pd.DataFrame:
        return self._pdf


class _DummySparkReader:
    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf

    def parquet(self, _uri: str) -> _DummySparkDataFrame:
        return _DummySparkDataFrame(self._pdf)


class _DummySpark:
    def __init__(self, pdf: pd.DataFrame):
        self.read = _DummySparkReader(pdf)


class _PredictorWithProba:
    def __init__(self):
        self.predict_calls = 0
        self.predict_proba_calls = 0

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        self.predict_calls += 1
        return pd.DataFrame({"prediction": np.zeros(len(x), dtype=int)})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        self.predict_proba_calls += 1
        return pd.DataFrame(
            {
                "proba_0": np.array([0.9, 0.2, 0.1, 0.8]),
                "proba_1": np.array([0.1, 0.8, 0.9, 0.2]),
            }
        )


def test_evaluate_predictor_uses_predict_for_label_metrics():
    predictor = _PredictorWithProba()
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    score = _evaluate_predictor(
        spark=_DummySpark(df),
        predictor=predictor,
        data_uri="unused",
        metric="accuracy",
        target_columns=["target"],
        group_column=None,
    )

    assert isinstance(score, float)
    assert predictor.predict_calls == 1
    assert predictor.predict_proba_calls == 0


def test_evaluate_predictor_uses_predict_proba_for_probability_metrics():
    predictor = _PredictorWithProba()
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 1, 0]})

    score = _evaluate_predictor(
        spark=_DummySpark(df),
        predictor=predictor,
        data_uri="unused",
        metric="roc_auc",
        target_columns=["target"],
        group_column=None,
    )

    assert isinstance(score, float)
    assert predictor.predict_calls == 0
    assert predictor.predict_proba_calls == 1


def test_evaluate_predictor_raises_when_probability_metric_missing_predict_proba():
    class _PredictOnly:
        def predict(self, x: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"prediction": np.zeros(len(x), dtype=int)})

    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 1, 0]})

    with pytest.raises(ValueError, match="does not implement predict_proba"):
        _evaluate_predictor(
            spark=_DummySpark(df),
            predictor=_PredictOnly(),
            data_uri="unused",
            metric="roc_auc",
            target_columns=["target"],
            group_column=None,
        )
