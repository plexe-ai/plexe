"""
This module defines the core interfaces for dataset handling in plexe.

The interfaces provide a consistent API for working with different types of datasets, regardless of their
underlying implementation. They define operations like splitting, sampling, and serialization that are common
across all dataset types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Type, TypeVar, Any, Literal, Dict, List

import numpy as np
import pandas as pd

# Type variable for the dataset interface
T = TypeVar("T", bound="Dataset")


@dataclass
class DatasetStructure:
    """
    Descriptor for the dataset structure.

    This descriptor provides metadata about the dataset's structure. The 'modality' field indicates the broad
    type of underlying data structure (e.g., tabular, tensor, etc.) in a framework-agnostic way (that is, it
    does not distinguish between PyTorch tensors and TensorFlow tensors, for example). The 'details' field
    contains the description of the dataset's structure, which can vary depending on the type of dataset.
    """

    # todo: expand this when more dataset types are added
    modality: Literal["table", "tensor", "other"] = field()
    features: List[str] = field()
    details: Dict[str, Any] = field()


class Dataset(ABC):
    """
    Base interface for all dataset implementations with universal operations.

    This interface defines operations that all dataset implementations must support, regardless of the underlying
    data format. These include splitting datasets into train/validation/test sets, sampling data, and serialization.
    """

    @abstractmethod
    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: str = None,
        random_state: int = None,
    ) -> Tuple[T, T, T]:
        """
        Split dataset into train, validation and test sets.

        :param train_ratio: Proportion of data to use for training
        :param val_ratio: Proportion of data to use for validation
        :param test_ratio: Proportion of data to use for testing
        :param stratify_column: Column to use for stratified splitting
        :param random_state: Random seed for reproducibility
        :returns: A tuple of (train_dataset, val_dataset, test_dataset)
        """
        pass

    @abstractmethod
    def sample(self, n: int = None, frac: float = None, replace: bool = False, random_state: int = None) -> T:
        """
        Sample records from dataset.

        :param n: Number of samples to take
        :param frac: Fraction of dataset to sample
        :param replace: Whether to sample with replacement
        :param random_state: Random seed for reproducibility
        :returns: A new dataset containing the sampled data
        """
        pass

    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Serialize dataset to bytes.

        :returns: Serialized dataset as bytes
        """
        pass

    @classmethod
    @abstractmethod
    def from_bytes(cls: Type[T], data: bytes) -> T:
        """
        Deserialize dataset from bytes.

        :param data: Serialized dataset as bytes
        :returns: Deserialized dataset
        """
        pass

    @property
    @abstractmethod
    def structure(self) -> DatasetStructure:
        """
        Return a descriptor of the dataset's structure.

        The structure descriptor has different details depending on the type of underlying dataset. This
        method is used to provide human-readable and LLM-readable information about the dataset's structure.

        :returns: Schema definition for the dataset
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        :returns: Number of items in the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Get an item from the dataset by index.

        :param index: Index of the item to retrieve
        :returns: The item at the specified index
        """
        pass


class TabularConvertible(ABC):
    """
    Interface for datasets that can be converted to tabular formats.

    This interface defines methods for converting a dataset to common tabular
    data formats like pandas DataFrames and numpy arrays.
    """

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        :returns: Dataset as pandas DataFrame
        """
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array.

        :returns: Dataset as numpy array
        """
        pass


class TorchConvertible(ABC):
    """
    Interface for datasets that can be converted to PyTorch formats.

    This interface defines methods for converting a dataset to PyTorch-specific
    data structures like Datasets and Tensors.
    """

    @abstractmethod
    def to_torch_dataset(self):
        """
        Convert to PyTorch Dataset.

        :returns: Dataset as PyTorch Dataset
        """
        pass

    @abstractmethod
    def to_torch_tensor(self):
        """
        Convert to PyTorch Tensor.

        :returns: Dataset as PyTorch Tensor
        """
        pass


class TensorflowConvertible(ABC):
    """
    Interface for datasets that can be converted to TensorFlow formats.

    This interface defines methods for converting a dataset to TensorFlow-specific
    data structures like tf.data.Dataset.
    """

    @abstractmethod
    def to_tf_dataset(self):
        """
        Convert to TensorFlow Dataset.

        :returns: Dataset as TensorFlow Dataset
        """
        pass
