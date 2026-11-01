"""Parquet shards must map every feature to the same model input position."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from plexe.utils.parquet_dataset import ParquetIterableDataset, parquet_batch_generator


@pytest.fixture
def shards(tmp_path):
    """Write shards with identical named features in different physical orders."""
    pq.write_table(pa.table({"left": [1.0, 2.0], "target": [10, 20], "right": [101.0, 102.0]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"right": [103.0, 104.0], "left": [3.0, 4.0], "target": [30, 40]}), tmp_path / "b.parquet")
    return tmp_path


@pytest.mark.parametrize("reader", ["batches", "torch"])
def test_shards_preserve_feature_positions(shards, reader):
    """Both streaming APIs must preserve the first shard's named column order."""
    if reader == "batches":
        batches = list(parquet_batch_generator(str(shards), "target", batch_size=1))
        features = np.concatenate([x for x, _ in batches])
        targets = np.concatenate([y for _, y in batches])
    else:
        pytest.importorskip("torch")
        rows = list(ParquetIterableDataset(str(shards), "target"))
        features = np.stack([x.numpy() for x, _ in rows])
        targets = np.concatenate([y.numpy() for _, y in rows])
    order = np.argsort(targets)
    np.testing.assert_array_equal(targets[order], [10, 20, 30, 40])
    np.testing.assert_array_equal(features[order], [[1, 101], [2, 102], [3, 103], [4, 104]])


@pytest.mark.parametrize("worker_id", [0, 1])
def test_workers_share_feature_order_even_when_assigned_different_shards(shards, monkeypatch, worker_id):
    """A worker's first assigned shard cannot define different input positions."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=worker_id, num_workers=2))
    rows = list(ParquetIterableDataset(str(shards), "target"))
    assert len(rows) == 2
    for features, target in rows:
        left = target.item() / 10
        np.testing.assert_array_equal(features.numpy(), [left, 100 + left])


@pytest.mark.parametrize("reader", ["batches", "torch"])
@pytest.mark.parametrize(
    "columns",
    [
        {"left": [5.0], "target": [50]},
        {"left": [5.0], "right": [105.0], "extra": [9.0], "target": [50]},
        {"left": [5.0], "right": [105.0]},
    ],
)
def test_inconsistent_shards_fail_instead_of_changing_model_inputs(shards, reader, columns):
    """Missing, extra and absent target columns cannot silently change inputs."""
    pq.write_table(pa.table(columns), shards / "c.parquet")
    with pytest.raises(ValueError, match="columns|target"):
        if reader == "batches":
            list(parquet_batch_generator(str(shards), "target"))
        else:
            pytest.importorskip("torch")
            list(ParquetIterableDataset(str(shards), "target"))


@pytest.mark.parametrize("reader", ["batches", "torch"])
def test_empty_first_shard_still_defines_feature_order(tmp_path, reader):
    """The schema, not the first nonempty batch or worker, defines positions."""
    pq.write_table(
        pa.table(
            {
                "left": pa.array([], type=pa.float32()),
                "target": pa.array([], type=pa.int64()),
                "right": pa.array([], type=pa.float32()),
            }
        ),
        tmp_path / "a.parquet",
    )
    pq.write_table(pa.table({"right": [103.0], "target": [30], "left": [3.0]}), tmp_path / "b.parquet")
    if reader == "batches":
        features = np.concatenate([x for x, _ in parquet_batch_generator(str(tmp_path), "target")])
    else:
        pytest.importorskip("torch")
        features = np.stack([x.numpy() for x, _ in ParquetIterableDataset(str(tmp_path), "target")])
    np.testing.assert_array_equal(features, [[3, 103]])
