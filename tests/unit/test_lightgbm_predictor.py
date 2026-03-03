"""
Unit tests for LightGBM predictor template.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from plexe.templates.inference.lightgbm_predictor import LightGBMPredictor


class DummyModel:
    """Minimal model stub with a predict method."""

    def predict(self, x):
        return np.zeros(len(x), dtype=int)


class DummyClassificationModel(DummyModel):
    """Minimal model stub with predict_proba for classification."""

    def predict_proba(self, x):
        return np.tile(np.array([[0.7, 0.3]]), (len(x), 1))


class DummyPipeline:
    """Minimal pipeline stub with a transform method."""

    def transform(self, x):
        return x


def _write_artifacts(base_dir: Path, with_encoder: bool = False) -> Path:
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(DummyModel(), artifacts_dir / "model.pkl")
    joblib.dump(DummyPipeline(), artifacts_dir / "pipeline.pkl")

    if with_encoder:
        encoder = LabelEncoder()
        encoder.fit(["no", "yes"])
        joblib.dump(encoder, artifacts_dir / "label_encoder.pkl")

    return artifacts_dir


def _write_metadata(base_dir: Path, task_type: str) -> None:
    artifacts_dir = base_dir / "artifacts"
    metadata_path = artifacts_dir / "metadata.json"
    metadata_path.write_text(f'{{"task_type": "{task_type}"}}', encoding="utf-8")


def test_lightgbm_predictor_basic(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    predictor = LightGBMPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    predictions = predictor.predict(input_df)

    assert list(predictions.columns) == ["prediction"]
    assert len(predictions) == 2


def test_lightgbm_predictor_label_encoder(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, with_encoder=True)

    predictor = LightGBMPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    predictions = predictor.predict(input_df)["prediction"].tolist()

    assert predictions == ["no", "no"]


def test_lightgbm_predictor_predict_proba_classification(tmp_path: Path) -> None:
    artifacts_dir = _write_artifacts(tmp_path)
    joblib.dump(DummyClassificationModel(), artifacts_dir / "model.pkl")
    _write_metadata(tmp_path, "binary_classification")

    predictor = LightGBMPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    probabilities = predictor.predict_proba(input_df)

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_lightgbm_predictor_predict_proba_without_metadata(tmp_path: Path) -> None:
    artifacts_dir = _write_artifacts(tmp_path)
    joblib.dump(DummyClassificationModel(), artifacts_dir / "model.pkl")

    predictor = LightGBMPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    probabilities = predictor.predict_proba(input_df)

    assert list(probabilities.columns) == ["proba_0", "proba_1"]
    assert len(probabilities) == 2


def test_lightgbm_predictor_predict_proba_raises_for_regression(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    _write_metadata(tmp_path, "regression")

    predictor = LightGBMPredictor(str(tmp_path))
    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(ValueError, match="only valid for classification"):
        predictor.predict_proba(input_df)
