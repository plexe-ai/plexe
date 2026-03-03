"""Unit tests for XGBoost predictor template."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from plexe.templates.inference.xgboost_predictor import XGBoostPredictor


class DummyModel:
    """Minimal predictor stub for tests."""

    def predict(self, x):
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x):
        return np.tile(np.array([[0.6, 0.4]]), (len(x), 1))


class DummyPipeline:
    """Minimal pipeline stub for tests."""

    def transform(self, x):
        return x


def _write_artifacts(base_dir: Path, task_type: str) -> None:
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(DummyModel(), artifacts_dir / "model.pkl")
    joblib.dump(DummyPipeline(), artifacts_dir / "pipeline.pkl")
    (artifacts_dir / "metadata.json").write_text(f'{{"task_type": "{task_type}"}}', encoding="utf-8")


def test_xgboost_predictor_predict_proba_classification(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, "binary_classification")

    predictor = XGBoostPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    probabilities = predictor.predict_proba(input_df)

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_xgboost_predictor_predict_proba_without_metadata(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, "binary_classification")
    (tmp_path / "artifacts" / "metadata.json").unlink()

    predictor = XGBoostPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    probabilities = predictor.predict_proba(input_df)

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_xgboost_predictor_predict_proba_raises_for_regression(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, "regression")

    predictor = XGBoostPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(ValueError, match="only valid for classification"):
        predictor.predict_proba(input_df)
