"""
This module provides the TabularDataset implementation, which handles tabular data like pandas DataFrames
and numpy arrays. It implements the DatasetInterface and provides methods for splitting, sampling, and
serialization of tabular data.
"""

import io
import pandas as pd
from typing import Tuple, Optional, Any

import numpy as np
from plexe.internal.common.datasets.interface import Dataset, TabularConvertible, DatasetStructure


class TabularDataset(Dataset, TabularConvertible):
    """
    Dataset implementation for tabular data.

    TabularDataset wraps pandas DataFrames and provides methods for common dataset operations like splitting,
    sampling, and serialization. It also implements the TabularConvertible interface to allow conversion to
    pandas and numpy formats.
    """

    def __init__(self, data: pd.DataFrame):
        self._data = self._validate(data)

    @staticmethod
    def _validate(data: Any) -> pd.DataFrame:
        """Ensure that the input is a pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        raise ValueError(f"TabularDataset only supports pandas DataFrame input, got {type(data)}")

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Tuple["TabularDataset", "TabularDataset", "TabularDataset"]:
        """
        Split dataset into train, validation and test sets.

        :param train_ratio: Proportion of data to use for training
        :param val_ratio: Proportion of data to use for validation
        :param test_ratio: Proportion of data to use for testing
        :param stratify_column: Column to use for stratified splitting
        :param random_state: Random seed for reproducibility
        :returns: A tuple of (train_dataset, val_dataset, test_dataset)
        :raises ValueError: If ratios don't sum to approximately 1.0
        """
        from sklearn.model_selection import train_test_split

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Split ratios must sum to 1.0")

        # Handle all-data-to-train edge case
        if val_ratio < 1e-10 and test_ratio < 1e-10:
            return (
                TabularDataset(self._data),
                TabularDataset(pd.DataFrame(columns=self._data.columns)),
                TabularDataset(pd.DataFrame(columns=self._data.columns)),
            )
        elif val_ratio < 1e-10:
            train_data, test_data = train_test_split(
                self._data,
                test_size=test_ratio / (train_ratio + test_ratio),
                stratify=self._data[stratify_column] if stratify_column else None,
                random_state=random_state,
            )
            return (
                TabularDataset(train_data),
                TabularDataset(pd.DataFrame(columns=self._data.columns)),
                TabularDataset(test_data),
            )
        elif test_ratio < 1e-10:
            train_data, val_data = train_test_split(
                self._data,
                test_size=val_ratio / (train_ratio + val_ratio),
                stratify=self._data[stratify_column] if stratify_column else None,
                random_state=random_state,
            )
            return (
                TabularDataset(train_data),
                TabularDataset(val_data),
                TabularDataset(pd.DataFrame(columns=self._data.columns)),
            )

        # Standard 3-way split
        train_data, temp_data = train_test_split(
            self._data,
            test_size=(val_ratio + test_ratio),
            stratify=self._data[stratify_column] if stratify_column else None,
            random_state=random_state,
        )
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_data[stratify_column] if stratify_column else None,
            random_state=random_state,
        )
        return (TabularDataset(train_data), TabularDataset(val_data), TabularDataset(test_data))

    def sample(
        self, n: int = None, frac: float = None, replace: bool = False, random_state: int = None
    ) -> "TabularDataset":
        """
        Sample records from dataset.

        :param n: number of samples to take
        :param frac: fraction of dataset to sample
        :param replace: whether to sample with replacement
        :param random_state: random seed for reproducibility
        :return: a new dataset containing the sampled data
        """
        return TabularDataset(self._data.sample(n=n, frac=frac, replace=replace, random_state=random_state))

    def to_bytes(self) -> bytes:
        """
        Serialize the dataset to bytes using Parquet format.

        :return: byte representation of the dataset
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            buffer = io.BytesIO()
            table = pa.Table.from_pandas(self._data)
            pq.write_table(table, buffer)
            return buffer.getvalue()
        except Exception as e:
            raise RuntimeError("Failed to serialize DataFrame to bytes") from e

    @classmethod
    def from_bytes(cls, data: bytes) -> "TabularDataset":
        """
        Deserialize bytes back into a TabularDataset.

        :param data: byte representation of dataset
        :return: TabularDataset instance
        """
        try:
            import pyarrow.parquet as pq

            buffer = io.BytesIO(data)
            table = pq.read_table(buffer)
            return cls(table.to_pandas())
        except Exception as e:
            raise RuntimeError("Failed to deserialize bytes to DataFrame") from e

    @property
    def structure(self) -> DatasetStructure:
        """
        Return structural metadata for the dataset.

        :return: DatasetStructure object describing features and shape
        """
        return DatasetStructure(
            modality="table",
            features=list(self._data.columns),
            details={
                "num_rows": len(self._data),
                "num_columns": self._data.shape[1],
                "column_names": list(self._data.columns),
                "column_types": self._data.dtypes.astype(str).to_dict(),
            },
        )

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the dataset as a pandas DataFrame."""
        return self._data

    def to_numpy(self) -> np.ndarray:
        """Convert the dataset to a NumPy array."""
        return self._data.to_numpy()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Any) -> Any:
        return self._data.iloc[item]
