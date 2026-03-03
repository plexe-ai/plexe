"""Unit tests for Keras predictor template semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plexe.templates.inference.keras_predictor import KerasPredictor


class DummyPipeline:
    """Minimal pipeline stub for tests."""

    def transform(self, x):
        return x


class DummyModel:
    """Minimal model stub for tests."""

    def __init__(self, output):
        self._output = output

    def predict(self, x, verbose=0):
        return self._output


def test_keras_probabilities_from_binary_logits() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = "binary_classification"
    predictor._loss_class = "BinaryCrossentropy"
    predictor._loss_config = {"from_logits": True}

    probs = predictor._probabilities_from_raw(np.array([[-2.0], [0.0], [2.0]]))

    assert probs.shape == (3, 2)
    assert np.allclose(probs[:, 0] + probs[:, 1], np.ones(3))
    assert probs[0, 1] < probs[1, 1] < probs[2, 1]


def test_keras_probabilities_from_binary_two_logit_output() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = "binary_classification"
    predictor._loss_class = "BinaryCrossentropy"
    predictor._loss_config = {"from_logits": True}

    probs = predictor._probabilities_from_raw(np.array([[2.0, -2.0], [-2.0, 2.0]]))

    assert probs.shape == (2, 2)
    assert np.isclose(probs[0].sum(), 1.0)
    assert np.isclose(probs[1].sum(), 1.0)
    assert probs[0, 0] > probs[0, 1]
    assert probs[1, 1] > probs[1, 0]


def test_keras_probabilities_from_multiclass_logits() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = "multiclass_classification"
    predictor._loss_class = "SparseCategoricalCrossentropy"
    predictor._loss_config = {"from_logits": True}

    probs = predictor._probabilities_from_raw(np.array([[1.0, 2.0, 3.0]]))

    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0)
    assert np.argmax(probs, axis=1)[0] == 2


def test_keras_probabilities_infer_logits_when_loss_config_missing() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = "binary_classification"
    predictor._loss_class = "BinaryCrossentropy"
    predictor._loss_config = {}

    probs = predictor._probabilities_from_raw(np.array([[-2.0], [2.0]]))

    assert probs.shape == (2, 2)
    assert np.allclose(probs[:, 0] + probs[:, 1], np.ones(2))
    assert probs[0, 1] < probs[1, 1]


def test_keras_predict_proba_raises_for_regression() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = "regression"
    predictor._loss_class = "MeanSquaredError"
    predictor._loss_config = {}
    predictor.pipeline = DummyPipeline()
    predictor.model = DummyModel(np.array([[0.5], [0.7]]))

    with pytest.raises(ValueError, match="only valid for classification"):
        predictor.predict_proba(pd.DataFrame({"f1": [1.0, 2.0]}))


def test_keras_predict_proba_allows_missing_task_metadata() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = ""
    predictor._loss_class = ""
    predictor._loss_config = {}
    predictor.pipeline = DummyPipeline()
    predictor.model = DummyModel(np.array([[-2.0], [2.0]]))

    probabilities = predictor.predict_proba(pd.DataFrame({"f1": [1.0, 2.0]}))

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2
    assert probabilities.iloc[0]["proba_1"] < probabilities.iloc[1]["proba_1"]


def test_keras_predict_proba_raises_on_non_finite_outputs() -> None:
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor._task_type = ""
    predictor._loss_class = ""
    predictor._loss_config = {}
    predictor.pipeline = DummyPipeline()
    predictor.model = DummyModel(np.array([[np.nan], [np.inf]]))

    with pytest.raises(ValueError, match="contain NaN/Inf"):
        predictor.predict_proba(pd.DataFrame({"f1": [1.0, 2.0]}))
