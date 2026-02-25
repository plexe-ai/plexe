"""
Unit tests for LightGBM predictor template.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from plexe.templates.inference.lightgbm_predictor import LightGBMPredictor


class DummyModel:
    """Minimal model stub with a predict method."""

    def predict(self, x):
        return np.zeros(len(x), dtype=int)


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
