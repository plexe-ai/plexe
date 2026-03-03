"""Unit tests for split URI submission validation."""

from __future__ import annotations

import pytest

from plexe.models import BuildContext
from plexe.tools.submission import get_save_split_uris_tool


class _DummySparkFrame:
    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count


class _DummySparkReader:
    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def parquet(self, uri: str) -> _DummySparkFrame:
        if uri not in self._counts:
            raise ValueError(f"Unknown URI: {uri}")
        return _DummySparkFrame(self._counts[uri])


class _DummySpark:
    def __init__(self, counts: dict[str, int]):
        self.read = _DummySparkReader(counts)


def _make_context(tmp_path) -> BuildContext:
    return BuildContext(
        user_id="user",
        experiment_id="exp",
        dataset_uri="dataset.parquet",
        work_dir=tmp_path,
        intent="predict transported",
    )


def test_save_split_uris_requires_test_when_expected(tmp_path):
    context = _make_context(tmp_path)
    spark = _DummySpark({"train_uri": 80, "val_uri": 20})

    save_split_uris = get_save_split_uris_tool(
        context=context,
        spark=spark,
        expected_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    )

    with pytest.raises(ValueError, match="non-empty test split is required"):
        save_split_uris(train_uri="train_uri", val_uri="val_uri")


def test_save_split_uris_canonicalizes_validation_key(tmp_path):
    context = _make_context(tmp_path)
    spark = _DummySpark({"train_uri": 80, "val_uri": 20})

    save_split_uris = get_save_split_uris_tool(
        context=context,
        spark=spark,
        expected_ratios={"train": 0.8, "validation": 0.2},
    )

    message = save_split_uris(train_uri="train_uri", val_uri="val_uri")

    assert "saved successfully" in message.lower()
    assert context.scratch["_train_uri"] == "train_uri"
    assert context.scratch["_val_uri"] == "val_uri"
    assert context.scratch["_test_uri"] is None
