"""
Tests for column exclusion pipeline.
"""

from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from plexe.constants import ScratchKeys
from plexe.models import BuildContext
from plexe.workflow import _exclude_problematic_columns


@pytest.fixture(scope="session")
def spark():
    spark_session = SparkSession.builder.master("local[1]").appName("plexe-column-exclusion-tests").getOrCreate()
    yield spark_session
    spark_session.stop()


def _write_parquet(spark: SparkSession, path: Path) -> str:
    df = spark.createDataFrame(
        [
            {"feature": 1, "leak": 10, "target": 0},
            {"feature": 2, "leak": 11, "target": 1},
        ]
    )
    df.write.mode("overwrite").parquet(str(path))
    return str(path)


def _make_context(tmp_path: Path, dataset_uri: str) -> BuildContext:
    return BuildContext(
        user_id="user-1",
        experiment_id="exp-1",
        dataset_uri=dataset_uri,
        work_dir=tmp_path,
        intent="predict target",
    )


def test_exclude_problematic_columns_drops_columns_and_returns_new_uri(spark, tmp_path):
    source_uri = _write_parquet(spark, tmp_path / "input.parquet")
    context = _make_context(tmp_path, source_uri)
    context.output_targets = ["target"]
    context.scratch[ScratchKeys.PROBLEMATIC_COLUMNS] = [
        {"column": "leak", "reason": "Correlation > 0.95", "category": "leakage"}
    ]

    filtered_uri = _exclude_problematic_columns(spark, source_uri, context, None)

    assert filtered_uri != source_uri
    filtered_df = spark.read.parquet(filtered_uri)
    assert "leak" not in filtered_df.columns
    assert context.excluded_columns == [{"column": "leak", "reason": "Correlation > 0.95"}]


def test_exclude_problematic_columns_noop_when_empty(spark, tmp_path):
    source_uri = _write_parquet(spark, tmp_path / "input.parquet")
    context = _make_context(tmp_path, source_uri)
    context.output_targets = ["target"]
    context.scratch[ScratchKeys.PROBLEMATIC_COLUMNS] = []

    filtered_uri = _exclude_problematic_columns(spark, source_uri, context, None)

    assert filtered_uri == source_uri
    assert context.excluded_columns == []


def test_build_context_round_trip_with_excluded_columns(tmp_path):
    context = BuildContext(
        user_id="user-2",
        experiment_id="exp-2",
        dataset_uri="/tmp/data.parquet",
        work_dir=tmp_path,
        intent="predict target",
        excluded_columns=[{"column": "leak", "reason": "leakage"}],
    )

    serialized = context.to_dict()
    restored = BuildContext.from_dict(serialized)

    assert restored.excluded_columns == context.excluded_columns


def test_exclude_problematic_columns_never_drops_target(spark, tmp_path):
    source_uri = _write_parquet(spark, tmp_path / "input.parquet")
    context = _make_context(tmp_path, source_uri)
    context.output_targets = ["target"]
    context.scratch[ScratchKeys.PROBLEMATIC_COLUMNS] = [
        {"column": "target", "reason": "Post-hoc derivative", "category": "leakage"}
    ]

    filtered_uri = _exclude_problematic_columns(spark, source_uri, context, None)

    assert filtered_uri == source_uri
    filtered_df = spark.read.parquet(filtered_uri)
    assert "target" in filtered_df.columns
    assert context.excluded_columns == []


def test_exclude_problematic_columns_never_drops_primary_input(spark, tmp_path):
    source_uri = _write_parquet(spark, tmp_path / "input.parquet")
    context = _make_context(tmp_path, source_uri)
    context.output_targets = ["target"]
    context.primary_input_column = "feature"
    context.scratch[ScratchKeys.PROBLEMATIC_COLUMNS] = [
        {"column": "feature", "reason": "Flagged by analysis", "category": "irrelevant"}
    ]

    filtered_uri = _exclude_problematic_columns(spark, source_uri, context, None)

    assert filtered_uri == source_uri
    filtered_df = spark.read.parquet(filtered_uri)
    assert "feature" in filtered_df.columns
    assert context.excluded_columns == []
