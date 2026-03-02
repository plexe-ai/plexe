"""Tests for resume-time model type filtering."""

from pathlib import Path

import pytest

from plexe.config import Config
from plexe.models import BuildContext
from plexe.workflow import _apply_allowed_model_types_on_resume


def _make_context(tmp_path: Path, viable_model_types: list[str]) -> BuildContext:
    return BuildContext(
        user_id="user-1",
        experiment_id="exp-1",
        dataset_uri="/tmp/data.parquet",
        work_dir=tmp_path,
        intent="predict churn",
        viable_model_types=viable_model_types,
    )


def test_filters_checkpoint_model_types_on_resume(tmp_path):
    context = _make_context(tmp_path, ["xgboost", "catboost", "lightgbm", "pytorch"])
    config = Config(allowed_model_types=["xgboost"])

    _apply_allowed_model_types_on_resume(context, config, start_phase=4)

    assert context.viable_model_types == ["xgboost"]


def test_uses_allowed_model_types_when_checkpoint_has_none(tmp_path):
    context = _make_context(tmp_path, [])
    config = Config(allowed_model_types=["pytorch"])

    _apply_allowed_model_types_on_resume(context, config, start_phase=4)

    assert context.viable_model_types == ["pytorch"]


def test_raises_when_allowed_types_do_not_intersect_checkpoint(tmp_path):
    context = _make_context(tmp_path, ["xgboost", "catboost"])
    config = Config(allowed_model_types=["pytorch"])

    with pytest.raises(ValueError, match="No model types remain"):
        _apply_allowed_model_types_on_resume(context, config, start_phase=4)


def test_does_not_filter_before_phase_one(tmp_path):
    context = _make_context(tmp_path, ["xgboost", "catboost"])
    config = Config(allowed_model_types=["xgboost"])

    _apply_allowed_model_types_on_resume(context, config, start_phase=1)

    assert context.viable_model_types == ["xgboost", "catboost"]
