"""Tests for streaming parquet data loading utilities."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from plexe.utils.parquet_dataset import (
    ParquetIterableDataset,
    get_dataset_size_bytes,
    get_parquet_feature_count,
    get_parquet_row_count,
    get_steps_per_epoch,
    parquet_batch_generator,
)

torch = pytest.importorskip("torch")


class TestMetadataUtilities:
    """Tests for parquet metadata helper functions."""

    def test_get_parquet_row_count(self, synthetic_parquet_classification):
        count = get_parquet_row_count(synthetic_parquet_classification["path"])
        assert count == synthetic_parquet_classification["n_rows"]

    def test_get_dataset_size_bytes_file(self, synthetic_parquet_classification):
        size = get_dataset_size_bytes(synthetic_parquet_classification["path"])
        assert size > 0

    def test_get_dataset_size_bytes_directory(self, tmp_path, synthetic_parquet_classification):
        # Create a directory with multiple parquet files
        subdir = tmp_path / "multi"
        subdir.mkdir()
        df = pd.read_parquet(synthetic_parquet_classification["path"])
        for i in range(3):
            pq.write_table(pa.Table.from_pandas(df), subdir / f"part_{i}.parquet")

        size = get_dataset_size_bytes(str(subdir))
        assert size > 0

    def test_get_dataset_size_bytes_nonexistent(self):
        size = get_dataset_size_bytes("/nonexistent/path")
        assert size == 0

    def test_get_parquet_feature_count(self, synthetic_parquet_classification):
        count = get_parquet_feature_count(
            synthetic_parquet_classification["path"],
            synthetic_parquet_classification["target_column"],
        )
        assert count == synthetic_parquet_classification["n_features"]

    def test_get_steps_per_epoch(self, synthetic_parquet_classification):
        steps = get_steps_per_epoch(synthetic_parquet_classification["path"], batch_size=32)
        expected = (synthetic_parquet_classification["n_rows"] + 31) // 32  # ceil division
        assert steps == expected


class TestParquetIterableDataset:
    """Tests for streaming iterable dataset behavior."""

    def test_yields_all_rows_classification(self, synthetic_parquet_classification):
        ds = ParquetIterableDataset(
            synthetic_parquet_classification["path"],
            target_column="target",
            task_type="multiclass_classification",
        )

        rows = list(ds)
        assert len(rows) == synthetic_parquet_classification["n_rows"]

        # Check shapes and dtypes
        x, y = rows[0]
        assert x.shape == (synthetic_parquet_classification["n_features"],)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
        assert y.ndim == 0  # scalar

    def test_yields_all_rows_regression(self, synthetic_parquet_regression):
        ds = ParquetIterableDataset(
            synthetic_parquet_regression["path"],
            target_column="target",
            task_type="regression",
        )

        rows = list(ds)
        assert len(rows) == synthetic_parquet_regression["n_rows"]

        x, y = rows[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert y.shape == (1,)  # [1] tensor, collates to [batch, 1]

    def test_yields_all_rows_binary(self, synthetic_parquet_classification):
        ds = ParquetIterableDataset(
            synthetic_parquet_classification["path"],
            target_column="target",
            task_type="binary_classification",
        )

        rows = list(ds)
        x, y = rows[0]
        assert y.dtype == torch.float32
        assert y.shape == (1,)  # [1] tensor, collates to [batch, 1]

    def test_directory_of_parquets(self, tmp_path, synthetic_parquet_classification):
        """Test loading from a directory containing multiple parquet files."""
        subdir = tmp_path / "parts"
        subdir.mkdir()
        df = pd.read_parquet(synthetic_parquet_classification["path"])

        # Split into 2 files
        mid = len(df) // 2
        pq.write_table(pa.Table.from_pandas(df.iloc[:mid]), subdir / "part_0.parquet", row_group_size=50)
        pq.write_table(pa.Table.from_pandas(df.iloc[mid:]), subdir / "part_1.parquet", row_group_size=50)

        ds = ParquetIterableDataset(str(subdir), target_column="target", task_type="multiclass_classification")
        rows = list(ds)
        assert len(rows) == len(df)

    def test_total_rows_property(self, synthetic_parquet_classification):
        ds = ParquetIterableDataset(
            synthetic_parquet_classification["path"],
            target_column="target",
            task_type="multiclass_classification",
        )
        assert ds.total_rows == synthetic_parquet_classification["n_rows"]

    def test_ddp_sharding(self, synthetic_parquet_classification):
        """Verify DDP sharding splits row groups across ranks."""
        path = synthetic_parquet_classification["path"]

        all_rows_by_rank = {}
        for rank in range(2):
            with (
                patch("torch.distributed.is_available", return_value=True),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_rank", return_value=rank),
                patch("torch.distributed.get_world_size", return_value=2),
            ):
                ds = ParquetIterableDataset(path, target_column="target", task_type="multiclass_classification")
                rows = list(ds)
                all_rows_by_rank[rank] = rows

        # Each rank gets a subset, together they cover all rows
        total = len(all_rows_by_rank[0]) + len(all_rows_by_rank[1])
        assert total == synthetic_parquet_classification["n_rows"]
        # Ranks should get different row groups (different counts due to 2 row groups)
        assert len(all_rows_by_rank[0]) > 0
        assert len(all_rows_by_rank[1]) > 0

    def test_feature_values_match_source(self, synthetic_parquet_classification):
        """Verify streamed data matches the original parquet content."""
        ds = ParquetIterableDataset(
            synthetic_parquet_classification["path"],
            target_column="target",
            task_type="multiclass_classification",
        )

        df = pd.read_parquet(synthetic_parquet_classification["path"])
        feature_cols = [c for c in df.columns if c != "target"]

        streamed_x = []
        streamed_y = []
        for x, y in ds:
            streamed_x.append(x.numpy())
            streamed_y.append(y.numpy())

        streamed_x = np.array(streamed_x)
        streamed_y = np.array(streamed_y)

        # Sort both by features to account for row-group shuffling
        expected_x = df[feature_cols].values
        expected_y = df["target"].values
        streamed_order = np.lexsort(streamed_x.T)
        expected_order = np.lexsort(expected_x.T)

        np.testing.assert_allclose(streamed_x[streamed_order], expected_x[expected_order], atol=1e-6)
        np.testing.assert_array_equal(streamed_y[streamed_order], expected_y[expected_order])


class TestParquetBatchGenerator:
    """Tests for Keras/TensorFlow parquet batch generator."""

    def test_yields_all_rows(self, synthetic_parquet_classification):
        total_rows = 0
        for X_batch, y_batch in parquet_batch_generator(
            synthetic_parquet_classification["path"],
            target_column="target",
            batch_size=64,
        ):
            assert X_batch.dtype == np.float32
            assert y_batch.dtype == np.float32
            assert X_batch.shape[1] == synthetic_parquet_classification["n_features"]
            total_rows += len(X_batch)

        assert total_rows == synthetic_parquet_classification["n_rows"]

    def test_batch_size_respected(self, synthetic_parquet_classification):
        batch_size = 50
        batches = list(
            parquet_batch_generator(
                synthetic_parquet_classification["path"],
                target_column="target",
                batch_size=batch_size,
            )
        )

        # All batches except possibly the last should be exactly batch_size
        for X_batch, _ in batches[:-1]:
            assert len(X_batch) == batch_size

    def test_directory_input(self, tmp_path, synthetic_parquet_classification):
        """Test generator with directory of parquet files."""
        subdir = tmp_path / "gen_parts"
        subdir.mkdir()
        df = pd.read_parquet(synthetic_parquet_classification["path"])
        mid = len(df) // 2
        pq.write_table(pa.Table.from_pandas(df.iloc[:mid]), subdir / "p0.parquet")
        pq.write_table(pa.Table.from_pandas(df.iloc[mid:]), subdir / "p1.parquet")

        total = sum(len(X) for X, _ in parquet_batch_generator(str(subdir), "target", batch_size=64))
        assert total == len(df)

    def test_values_match_source(self, synthetic_parquet_classification):
        """Verify batched data matches original parquet content."""
        df = pd.read_parquet(synthetic_parquet_classification["path"])
        feature_cols = [c for c in df.columns if c != "target"]

        all_x, all_y = [], []
        for X_batch, y_batch in parquet_batch_generator(
            synthetic_parquet_classification["path"],
            target_column="target",
            batch_size=1000,  # Single batch to get all data
        ):
            all_x.append(X_batch)
            all_y.append(y_batch)

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        np.testing.assert_allclose(all_x, df[feature_cols].values, atol=1e-6)
        np.testing.assert_allclose(all_y, df["target"].values.astype(np.float32), atol=1e-6)
