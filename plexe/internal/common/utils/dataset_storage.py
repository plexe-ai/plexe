"""
This module provides utilities for dataset storage and transfer across processes.

These utilities work with any class implementing the DatasetInterface and provide
functions for storing datasets to files, reading datasets from files, and using
shared memory for cross-process dataset sharing.
"""

from typing import Type, TypeVar, Optional
import logging

from plexe.internal.common.datasets.interface import Dataset

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Dataset)


def write_dataset_to_file(dataset: Dataset, path: str) -> None:
    """
    Write dataset to a file.

    :param dataset: The dataset to write to a file
    :param path: Path to write the dataset to
    """
    with open(path, "wb") as f:
        f.write(dataset.to_bytes())


def read_dataset_from_file(dataset_class: Type[T], path: str) -> T:
    """
    Read dataset from a file.

    :param dataset_class: The dataset class to instantiate
    :param path: Path to read the dataset from
    :returns: Instantiated dataset of the specified class
    """
    with open(path, "rb") as f:
        return dataset_class.from_bytes(f.read())


def dataset_to_shared_memory(dataset: Dataset, name: str) -> None:
    """
    Place dataset in shared memory for cross-process access.

    This function serializes a dataset and places it in shared memory
    with the given name, allowing other processes to access it.

    :param dataset: The dataset to place in shared memory
    :param name: Name of the shared memory segment
    :raises ImportError: If shared memory is not available
    """
    try:
        from multiprocessing import shared_memory

        # Serialize the dataset
        data = dataset.to_bytes()

        # Create shared memory segment
        shm = shared_memory.SharedMemory(name=name, create=True, size=len(data))

        # Copy dataset bytes to shared memory
        shm.buf[: len(data)] = data

        return shm
    except ImportError:
        raise ImportError("Shared memory requires Python 3.8+ and the multiprocessing module")


def dataset_from_shared_memory(dataset_class: Type[T], name: str, size: Optional[int] = None) -> T:
    """
    Retrieve dataset from shared memory.

    This function retrieves a serialized dataset from shared memory
    and deserializes it into an instance of the specified class.

    :param dataset_class: The dataset class to instantiate
    :param name: Name of the shared memory segment
    :param size: Size of the dataset in bytes (optional)
    :returns: Instantiated dataset of the specified class
    """
    try:
        from multiprocessing import shared_memory

        # Access the shared memory segment
        shm = shared_memory.SharedMemory(name=name)

        # Determine the size of the data
        if size is None:
            # Find the null terminator if size is not specified
            # This assumes the shared memory was filled with zeros initially
            # and data is not binary (contains no null bytes)
            for i in range(shm.size):
                if shm.buf[i] == 0:
                    size = i
                    break
            else:
                size = shm.size

        # Deserialize the dataset
        data = bytes(shm.buf[:size])
        return dataset_class.from_bytes(data)
    except ImportError:
        raise ImportError("Shared memory requires Python 3.8+ and the multiprocessing module")
