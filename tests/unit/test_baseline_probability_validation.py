"""Unit tests for baseline probability validation behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plexe.models import BuildContext, Metric
from plexe.tools.submission import get_validate_baseline_predictor_tool


def _make_context(tmp_path) -> BuildContext:
    context = BuildContext(
        user_id="user",
        experiment_id="exp",
        dataset_uri="/tmp/dataset.parquet",
        work_dir=tmp_path,
        intent="predict churn",
    )
    context.output_targets = ["target"]
    return context


def test_validate_baseline_predictor_requires_predict_proba_for_probability_metrics(tmp_path):
    context = _make_context(tmp_path)
    context.metric = Metric(name="roc_auc", optimization_direction="higher")
    val_df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 1, 0]})

    class HeuristicBaselinePredictor:
        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    validate_tool = get_validate_baseline_predictor_tool(context, val_df)
    predictor = HeuristicBaselinePredictor()

    with pytest.raises(ValueError, match="requires probability scores"):
        validate_tool(predictor, "baseline", "missing predict_proba")


def test_validate_baseline_predictor_accepts_predict_proba_for_probability_metrics(tmp_path):
    context = _make_context(tmp_path)
    context.metric = Metric(name="roc_auc", optimization_direction="higher")
    val_df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 1, 0]})

    class HeuristicBaselinePredictor:
        def predict(self, x):
            return np.zeros(len(x), dtype=int)

        def predict_proba(self, x):
            return pd.DataFrame(
                {
                    "proba_0": np.array([0.9, 0.2, 0.1, 0.8]),
                    "proba_1": np.array([0.1, 0.8, 0.9, 0.2]),
                }
            )

    validate_tool = get_validate_baseline_predictor_tool(context, val_df)
    predictor = HeuristicBaselinePredictor()
    message = validate_tool(predictor, "baseline", "has predict_proba")

    assert "validated" in message.lower()
    assert context.baseline_performance is not None
