"""
Unit tests for pipeline_runner feature name resolution.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plexe.templates.features.pipeline_runner import _resolve_transformed_feature_names


class NoFeatureNamesTransformer(BaseEstimator, TransformerMixin):
    """Transformer without get_feature_names_out."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x


class SelectFirstColumnTransformer(BaseEstimator, TransformerMixin):
    """Transformer that reduces output to a single column."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if hasattr(x, "iloc"):
            return x.iloc[:, [0]]
        return x[:, [0]]


def test_resolve_feature_names_uses_pipeline_minus_last():
    """Falls back to pipeline[:-1] when last step lacks get_feature_names_out."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    pipeline = Pipeline([("scale", StandardScaler()), ("noop", NoFeatureNamesTransformer())])

    pipeline.fit(df)

    names, source = _resolve_transformed_feature_names(
        fitted_pipeline=pipeline,
        feature_columns=["a", "b"],
        num_output_features=2,
    )

    assert names == ["a", "b"]
    assert "pipeline[:-1]" in source


def test_resolve_feature_names_falls_back_on_mismatch():
    """Returns generic names when resolved names don't match output count."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    pipeline = Pipeline([("scale", StandardScaler()), ("select", SelectFirstColumnTransformer())])

    pipeline.fit(df)

    names, source = _resolve_transformed_feature_names(
        fitted_pipeline=pipeline,
        feature_columns=["a", "b"],
        num_output_features=1,
    )

    assert names == ["feature_0"]
    assert source == "generic_mismatch"


def test_resolve_feature_names_falls_back_when_unavailable():
    """Returns generic names when no get_feature_names_out is available."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    pipeline = Pipeline([("noop", NoFeatureNamesTransformer())])

    pipeline.fit(df)

    names, source = _resolve_transformed_feature_names(
        fitted_pipeline=pipeline,
        feature_columns=["a", "b"],
        num_output_features=2,
    )

    assert names == ["feature_0", "feature_1"]
    assert source == "generic"
