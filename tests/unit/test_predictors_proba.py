"""
Tests for predictor predict_proba implementations.
"""

import numpy as np
import pandas as pd
import pytest

from plexe.templates.inference.keras_predictor import KerasPredictor
from plexe.templates.inference.lightgbm_predictor import LightGBMPredictor
from plexe.templates.inference.xgboost_predictor import XGBoostPredictor


class _DummyPipeline:
    def transform(self, x: pd.DataFrame):
        return x


class _DummyEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)


class _DummyProbaModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, x):
        return self._proba


def _assert_proba_df(df: pd.DataFrame, expected_rows: int, expected_cols: int):
    assert df.shape == (expected_rows, expected_cols)
    row_sums = df.sum(axis=1).values
    assert np.allclose(row_sums, 1.0, atol=1e-6)


@pytest.mark.parametrize("predictor_cls", [XGBoostPredictor, LightGBMPredictor])
def test_tree_predictors_predict_proba_binary(predictor_cls):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    predictor = predictor_cls.__new__(predictor_cls)
    predictor.model = _DummyProbaModel(proba)
    predictor.pipeline = _DummyPipeline()
    predictor.label_encoder = _DummyEncoder(["no", "yes"])
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}))

    assert list(df.columns) == ["proba_no", "proba_yes"]
    _assert_proba_df(df, expected_rows=2, expected_cols=2)


@pytest.mark.parametrize("predictor_cls", [XGBoostPredictor, LightGBMPredictor])
def test_tree_predictors_predict_proba_multiclass(predictor_cls):
    proba = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    predictor = predictor_cls.__new__(predictor_cls)
    predictor.model = _DummyProbaModel(proba)
    predictor.pipeline = _DummyPipeline()
    predictor.label_encoder = _DummyEncoder(["class_a", "class_b", "class_c"])
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}))

    assert list(df.columns) == ["proba_class_a", "proba_class_b", "proba_class_c"]
    _assert_proba_df(df, expected_rows=2, expected_cols=3)


def test_catboost_predictor_predict_proba_binary():
    pytest.importorskip("catboost")
    from plexe.templates.inference.catboost_predictor import CatBoostPredictor

    proba = np.array([[0.1, 0.9], [0.7, 0.3]])
    predictor = CatBoostPredictor.__new__(CatBoostPredictor)
    predictor.model = _DummyProbaModel(proba)
    predictor.pipeline = _DummyPipeline()
    predictor.label_encoder = _DummyEncoder(["neg", "pos"])
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}))

    assert list(df.columns) == ["proba_neg", "proba_pos"]
    _assert_proba_df(df, expected_rows=2, expected_cols=2)


def test_catboost_predictor_predict_proba_multiclass():
    pytest.importorskip("catboost")
    from plexe.templates.inference.catboost_predictor import CatBoostPredictor

    proba = np.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])
    predictor = CatBoostPredictor.__new__(CatBoostPredictor)
    predictor.model = _DummyProbaModel(proba)
    predictor.pipeline = _DummyPipeline()
    predictor.label_encoder = _DummyEncoder(["neg", "neutral", "pos"])
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}))

    assert list(df.columns) == ["proba_neg", "proba_neutral", "proba_pos"]
    _assert_proba_df(df, expected_rows=2, expected_cols=3)


def test_keras_predictor_predict_proba_binary():
    class DummyKerasModel:
        def __init__(self, outputs):
            self._outputs = outputs

        def predict(self, x, verbose=0):
            return self._outputs

    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor.model = DummyKerasModel(np.array([[0.2], [0.8]]))
    predictor.pipeline = _DummyPipeline()
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2]}))

    assert list(df.columns) == ["proba_0", "proba_1"]
    _assert_proba_df(df, expected_rows=2, expected_cols=2)


def test_keras_predictor_predict_proba_multiclass():
    class DummyKerasModel:
        def __init__(self, outputs):
            self._outputs = outputs

        def predict(self, x, verbose=0):
            return self._outputs

    outputs = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3]])
    predictor = KerasPredictor.__new__(KerasPredictor)
    predictor.model = DummyKerasModel(outputs)
    predictor.pipeline = _DummyPipeline()
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2]}))

    assert list(df.columns) == ["proba_0", "proba_1", "proba_2"]
    _assert_proba_df(df, expected_rows=2, expected_cols=3)


def test_pytorch_predictor_predict_proba_binary():
    torch = pytest.importorskip("torch")
    from plexe.templates.inference.pytorch_predictor import PyTorchPredictor

    class DummyTorchModel:
        def __call__(self, x):
            return torch.tensor([[1.0, 2.0], [2.5, 0.5]], dtype=torch.float32)

    predictor = PyTorchPredictor.__new__(PyTorchPredictor)
    predictor.model = DummyTorchModel()
    predictor.pipeline = _DummyPipeline()
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2]}))

    assert list(df.columns) == ["proba_0", "proba_1"]
    _assert_proba_df(df, expected_rows=2, expected_cols=2)


def test_pytorch_predictor_predict_proba_multiclass():
    torch = pytest.importorskip("torch")
    from plexe.templates.inference.pytorch_predictor import PyTorchPredictor

    class DummyTorchModel:
        def __call__(self, x):
            return torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]], dtype=torch.float32)

    predictor = PyTorchPredictor.__new__(PyTorchPredictor)
    predictor.model = DummyTorchModel()
    predictor.pipeline = _DummyPipeline()
    predictor.task_type = "classification"

    df = predictor.predict_proba(pd.DataFrame({"f1": [1, 2]}))

    assert list(df.columns) == ["proba_0", "proba_1", "proba_2"]
    _assert_proba_df(df, expected_rows=2, expected_cols=3)


@pytest.mark.parametrize("predictor_cls", [XGBoostPredictor, KerasPredictor])
def test_predict_proba_raises_for_non_classification(predictor_cls):
    predictor = predictor_cls.__new__(predictor_cls)
    predictor.task_type = "regression"
    predictor.pipeline = _DummyPipeline()

    class DummyModel:
        def predict_proba(self, x):
            return np.array([[0.1, 0.9]])

        def predict(self, x, verbose=0):
            return np.array([[0.1], [0.9]])

    predictor.model = DummyModel()

    with pytest.raises(NotImplementedError, match="only available for classification models"):
        predictor.predict_proba(pd.DataFrame({"f1": [1, 2]}))
