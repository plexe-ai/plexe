"""Unit tests for PyTorch predictor template semantics."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from plexe.templates.inference.pytorch_predictor import PyTorchPredictor


class DummyPipeline:
    """Minimal pipeline stub for tests."""

    def transform(self, x):
        return x.values


class DummyModel:
    """Minimal callable model stub for tests."""

    def __init__(self, outputs):
        self._outputs = outputs

    def __call__(self, x_tensor):
        return self._outputs


def test_pytorch_predict_proba_binary_classification() -> None:
    predictor = PyTorchPredictor.__new__(PyTorchPredictor)
    predictor._task_type = "binary_classification"
    predictor.pipeline = DummyPipeline()
    predictor.model = DummyModel(torch.tensor([[-2.0], [2.0]], dtype=torch.float32))

    probabilities = predictor.predict_proba(pd.DataFrame({"f1": [0.0, 1.0]}))

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2
    assert probabilities.iloc[0]["proba_1"] < probabilities.iloc[1]["proba_1"]


def test_pytorch_predict_proba_allows_missing_task_metadata() -> None:
    predictor = PyTorchPredictor.__new__(PyTorchPredictor)
    predictor._task_type = ""
    predictor.pipeline = DummyPipeline()
    predictor.model = DummyModel(torch.tensor([[-2.0], [2.0]], dtype=torch.float32))

    probabilities = predictor.predict_proba(pd.DataFrame({"f1": [0.0, 1.0]}))

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_pytorch_predict_proba_raises_for_regression() -> None:
    predictor = PyTorchPredictor.__new__(PyTorchPredictor)
    predictor._task_type = "regression"
    predictor.pipeline = DummyPipeline()
    predictor.model = DummyModel(torch.tensor([[0.2], [0.8]], dtype=torch.float32))

    with pytest.raises(ValueError, match="only valid for classification"):
        predictor.predict_proba(pd.DataFrame({"f1": [0.0, 1.0]}))
