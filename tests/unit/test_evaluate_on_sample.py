"""
Tests for evaluate_on_sample probability handling.
"""

from pathlib import Path

import pandas as pd
import pytest

from plexe import helpers
from plexe.config import ModelType
from plexe.helpers import evaluate_on_sample


class _DummyParquet:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def toPandas(self):
        return self._df


class _DummyReader:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def parquet(self, uri: str):
        return _DummyParquet(self._df)


class _DummySpark:
    def __init__(self, df: pd.DataFrame):
        self.read = _DummyReader(df)


def test_evaluate_on_sample_uses_proba(monkeypatch):
    sample_df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    spark = _DummySpark(sample_df)

    class DummyPredictor:
        def __init__(self, model_dir: str):
            pass

        def predict(self, x):
            return pd.DataFrame({"prediction": [0, 0, 0]})

        def predict_proba(self, x):
            return pd.DataFrame({"proba_0": [0.8, 0.2, 0.7], "proba_1": [0.2, 0.8, 0.3]})

    import plexe.templates.inference.xgboost_predictor as xgb_module

    monkeypatch.setattr(xgb_module, "XGBoostPredictor", DummyPredictor)
    monkeypatch.setattr(helpers, "compute_metric_proba", lambda *args, **kwargs: 0.77)
    monkeypatch.setattr(helpers, "compute_metric", lambda *args, **kwargs: 0.11)

    result = evaluate_on_sample(
        spark=spark,
        sample_uri="dummy",
        model_artifacts_path=Path("."),
        model_type=ModelType.XGBOOST,
        metric="roc_auc",
        target_columns=["target"],
    )

    assert result == 0.77


def test_evaluate_on_sample_falls_back_without_proba(monkeypatch):
    sample_df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    spark = _DummySpark(sample_df)

    class DummyPredictor:
        def __init__(self, model_dir: str):
            pass

        def predict(self, x):
            return pd.DataFrame({"prediction": [0, 1, 0]})

    import plexe.templates.inference.xgboost_predictor as xgb_module

    monkeypatch.setattr(xgb_module, "XGBoostPredictor", DummyPredictor)
    monkeypatch.setattr(helpers, "compute_metric_proba", lambda *args, **kwargs: 0.77)
    monkeypatch.setattr(helpers, "compute_metric", lambda *args, **kwargs: 0.11)

    with pytest.raises(ValueError, match="requires probability outputs"):
        evaluate_on_sample(
            spark=spark,
            sample_uri="dummy",
            model_artifacts_path=Path("."),
            model_type=ModelType.XGBOOST,
            metric="roc_auc",
            target_columns=["target"],
        )


def test_evaluate_on_sample_uses_label_path_for_label_metrics(monkeypatch):
    sample_df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    spark = _DummySpark(sample_df)

    class DummyPredictor:
        def __init__(self, model_dir: str):
            pass

        def predict(self, x):
            return pd.DataFrame({"prediction": [0, 1, 0]})

        def predict_proba(self, x):
            raise AssertionError("predict_proba should not be called for label metrics")

    import plexe.templates.inference.xgboost_predictor as xgb_module

    monkeypatch.setattr(xgb_module, "XGBoostPredictor", DummyPredictor)
    monkeypatch.setattr(helpers, "compute_metric", lambda *args, **kwargs: 0.42)

    result = evaluate_on_sample(
        spark=spark,
        sample_uri="dummy",
        model_artifacts_path=Path("."),
        model_type=ModelType.XGBOOST,
        metric="accuracy",
        target_columns=["target"],
    )

    assert result == 0.42
