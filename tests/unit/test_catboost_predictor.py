"""Unit tests for CatBoost predictor template."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

catboost = pytest.importorskip("catboost")

from plexe.templates.inference.catboost_predictor import CatBoostPredictor


class DummyPipeline:
    """Minimal pipeline stub for tests."""

    def transform(self, x):
        return x


def test_catboost_predictor_predict_proba_classification() -> None:
    model = catboost.CatBoostClassifier(iterations=5, verbose=False)
    X_train = pd.DataFrame({"f1": [0.0, 0.1, 0.9, 1.0], "f2": [0.0, 0.2, 0.8, 1.0]})
    y_train = np.array([0, 0, 1, 1])
    model.fit(X_train, y_train)

    predictor = CatBoostPredictor.__new__(CatBoostPredictor)
    predictor._task_type = "binary_classification"
    predictor.model = model
    predictor.pipeline = DummyPipeline()

    probabilities = predictor.predict_proba(pd.DataFrame({"f1": [0.05, 0.95], "f2": [0.05, 0.95]}))

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_catboost_predictor_predict_proba_allows_missing_task_metadata() -> None:
    model = catboost.CatBoostClassifier(iterations=5, verbose=False)
    X_train = pd.DataFrame({"f1": [0.0, 0.1, 0.9, 1.0], "f2": [0.0, 0.2, 0.8, 1.0]})
    y_train = np.array([0, 0, 1, 1])
    model.fit(X_train, y_train)

    predictor = CatBoostPredictor.__new__(CatBoostPredictor)
    predictor._task_type = ""
    predictor.model = model
    predictor.pipeline = DummyPipeline()

    probabilities = predictor.predict_proba(pd.DataFrame({"f1": [0.05, 0.95], "f2": [0.05, 0.95]}))

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_catboost_predictor_predict_proba_raises_for_regression() -> None:
    predictor = CatBoostPredictor.__new__(CatBoostPredictor)
    predictor._task_type = "regression"
    predictor.model = object()
    predictor.pipeline = DummyPipeline()

    with pytest.raises(ValueError, match="only valid for classification"):
        predictor.predict_proba(pd.DataFrame({"f1": [0.1], "f2": [0.2]}))
