"""
Unit tests for validation functions.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plexe.validation.validators import (
    validate_sklearn_pipeline,
    validate_xgboost_params,
    validate_model_definition,
)
from plexe.config import ModelType


# ============================================
# Pipeline Validation Tests
# ============================================


def test_validate_sklearn_pipeline_success():
    """Test valid pipeline passes validation."""
    pipeline = Pipeline([("scaler", StandardScaler())])
    sample_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})

    is_valid, error = validate_sklearn_pipeline(pipeline, sample_df, target_columns=["target"])

    assert is_valid
    assert error == ""


def test_validate_sklearn_pipeline_wrong_type():
    """Test non-pipeline fails validation."""
    not_a_pipeline = {"scaler": StandardScaler()}
    sample_df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})

    is_valid, error = validate_sklearn_pipeline(not_a_pipeline, sample_df, target_columns=["target"])

    assert not is_valid
    assert "must be sklearn.pipeline.pipeline" in error.lower()


# ============================================
# XGBoost Validation Tests
# ============================================


def test_validate_xgboost_params_success():
    """Test valid XGBoost params pass validation."""
    params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1}

    is_valid, error = validate_xgboost_params(params)

    assert is_valid
    assert error == ""


def test_validate_xgboost_params_invalid_max_depth():
    """Test invalid max_depth fails validation."""
    params = {"max_depth": -1}

    is_valid, error = validate_xgboost_params(params)

    assert not is_valid
    assert "max_depth" in error


def test_validate_xgboost_params_invalid_type():
    """Test non-dict fails validation."""
    params = "not a dict"

    is_valid, error = validate_xgboost_params(params)

    assert not is_valid
    assert "must be dict" in error


# ============================================
# Model Definition Validation Tests
# ============================================


def test_validate_model_definition_xgboost():
    """Test XGBoost model definition validation."""
    definition = {"max_depth": 3, "n_estimators": 50}

    is_valid, error = validate_model_definition(ModelType.XGBOOST, definition)

    assert is_valid


def test_validate_model_definition_unknown_type():
    """Test unknown model type fails validation."""
    definition = {}

    is_valid, error = validate_model_definition("unknown", definition)

    assert not is_valid
