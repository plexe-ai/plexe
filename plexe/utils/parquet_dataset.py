"""
Streaming parquet data loading utilities for large-dataset training.

Reads parquet files lazily via PyArrow row groups instead of loading
everything into memory. Supports PyTorch DataLoader and Keras tf.data
integration, including DDP rank sharding and DataLoader worker sharding.
"""

import logging
import math
import random  # noqa: F401 - used in ParquetIterableDataset.__iter__
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

try:
    import torch
    import torch.distributed as dist
    import torch.utils.data

    _IterableDatasetBase = torch.utils.data.IterableDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    _IterableDatasetBase = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def get_parquet_row_count(uri: str) -> int:
    """Get total row count from parquet metadata without reading data."""
    total = 0
    for f in _resolve_parquet_files(uri):
        pf = pq.ParquetFile(f)
        total += pf.metadata.num_rows
    return total


def get_dataset_size_bytes(uri: str) -> int:
    """Get dataset size in bytes for a local file or directory of parquet files."""
    path = Path(uri)
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*.parquet"))
    return 0


def _resolve_parquet_files(uri: str) -> list[str]:
    """Resolve a URI to a list of parquet file paths.

    Handles both single files and directories containing parquet files.
    """
    path = Path(uri)
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = sorted(str(f) for f in path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {uri}")
        return files
    else:
        raise FileNotFoundError(f"Path does not exist: {uri}")


class ParquetIterableDataset(_IterableDatasetBase):
    """Streaming parquet dataset for PyTorch DataLoader.

    Reads parquet files row-group by row-group, yielding individual samples.
    Supports sharding across DDP ranks and DataLoader workers.

    Inherits from torch.utils.data.IterableDataset so PyTorch DataLoader
    knows to use the iterable protocol instead of map-style indexing.
    """

    def __init__(
        self,
        uri: str,
        target_column: str,
        task_type: str = "regression",
    ):
        """
        Args:
            uri: Path to parquet file or directory of parquet files
            target_column: Name of the target column
            task_type: One of "multiclass_classification", "binary_classification", "regression"
                       (legacy: "classification", "binary" also accepted)
        """
        super().__init__()
        self._files = _resolve_parquet_files(uri)
        self._target_column = target_column
        self._task_type = task_type

        # Build index of (file_idx, row_group_idx) pairs
        self._row_group_index: list[tuple[int, int]] = []
        self._total_rows = 0
        for file_idx, file_path in enumerate(self._files):
            pf = pq.ParquetFile(file_path)
            for rg_idx in range(pf.metadata.num_row_groups):
                self._row_group_index.append((file_idx, rg_idx))
            self._total_rows += pf.metadata.num_rows

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def _get_assigned_row_groups(self) -> list[tuple[int, int]]:
        """Determine which row groups this worker should process.

        Shards first by DDP rank, then by DataLoader worker.
        """
        rgs = list(self._row_group_index)

        # Shard across DDP ranks (balanced so all ranks get equal count)
        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if len(rgs) > 0:
                per_rank = math.ceil(len(rgs) / world_size)
                total = per_rank * world_size
                balanced = [rgs[i % len(rgs)] for i in range(total)]
                rgs = balanced[rank * per_rank : (rank + 1) * per_rank]

        # Shard across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rgs = rgs[worker_info.id :: worker_info.num_workers]

        return rgs

    def __iter__(self) -> Iterator:
        assigned_rgs = self._get_assigned_row_groups()
        random.shuffle(assigned_rgs)

        # Cache ParquetFile handles to avoid re-reading footers
        open_files: dict[int, pq.ParquetFile] = {}
        for file_idx, rg_idx in assigned_rgs:
            if file_idx not in open_files:
                open_files[file_idx] = pq.ParquetFile(self._files[file_idx])
            pf = open_files[file_idx]
            table = pf.read_row_group(rg_idx)
            df = table.to_pandas()

            feature_cols = [c for c in df.columns if c != self._target_column]
            X = df[feature_cols].values.astype(np.float32)
            y_raw = df[self._target_column].values

            # Yield individual samples for DataLoader batching
            for i in range(len(X)):
                x_tensor = torch.from_numpy(X[i].copy())
                if self._task_type in ("classification", "multiclass_classification"):
                    # CrossEntropyLoss expects scalar long targets -> batch becomes [batch]
                    y_tensor = torch.tensor(int(y_raw[i]), dtype=torch.long)
                else:
                    # BCEWithLogitsLoss / MSELoss expect [batch, 1] shape
                    # Yield [1] tensor so DataLoader collates to [batch, 1]
                    y_tensor = torch.tensor([float(y_raw[i])], dtype=torch.float32)
                yield x_tensor, y_tensor


def parquet_batch_generator(
    uri: str,
    target_column: str,
    batch_size: int = 1024,
    task_type: str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Streaming parquet batch generator for Keras/TensorFlow.

    Reads parquet file(s) in batches using PyArrow's iter_batches,
    yielding (X_batch, y_batch) numpy arrays suitable for
    tf.data.Dataset.from_generator().

    Args:
        uri: Path to parquet file or directory of parquet files
        target_column: Name of the target column
        batch_size: Number of rows per batch
        task_type: Canonical task type used to choose y dtype

    Yields:
        (features_array, target_array) tuples of numpy arrays
    """
    files = _resolve_parquet_files(uri)

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        columns = [c for c in pf.schema_arrow.names if c != target_column]

        for batch in pf.iter_batches(batch_size=batch_size, columns=columns + [target_column]):
            df = batch.to_pandas()
            X = df[columns].values.astype(np.float32)
            y_dtype = np.int64 if task_type == "multiclass_classification" else np.float32
            y = df[target_column].values.astype(y_dtype)
            yield X, y


def get_parquet_feature_count(uri: str, target_column: str) -> int:
    """Get number of feature columns (total columns minus target)."""
    files = _resolve_parquet_files(uri)
    pf = pq.ParquetFile(files[0])
    return len([c for c in pf.schema_arrow.names if c != target_column])


def get_steps_per_epoch(uri: str, batch_size: int) -> int:
    """Compute number of steps per epoch for a parquet dataset."""
    total_rows = 0
    files = _resolve_parquet_files(uri)
    for f in files:
        pf = pq.ParquetFile(f)
        total_rows += pf.metadata.num_rows
    return math.ceil(total_rows / batch_size)
