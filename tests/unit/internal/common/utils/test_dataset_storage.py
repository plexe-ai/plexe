"""
Tests for the dataset storage utilities.

This module verifies:
1. Writing datasets to files and reading them back
2. Storing datasets in shared memory and retrieving them
3. Error handling for invalid file paths and data
"""

import os
import pytest

from plexe.internal.common.datasets.tabular import TabularDataset
from plexe.internal.common.utils.dataset_storage import (
    write_dataset_to_file,
    read_dataset_from_file,
    dataset_to_shared_memory,
    dataset_from_shared_memory,
)
import pandas as pd


def test_write_and_read_file(tmp_path):
    """Test writing a dataset to a file and reading it back."""
    # Create a test dataset
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [0, 1, 0]})
    dataset = TabularDataset(df)

    # Write to file
    file_path = tmp_path / "test_dataset.bin"
    write_dataset_to_file(dataset, str(file_path))

    # Check that file exists
    assert file_path.exists()
    assert os.path.getsize(str(file_path)) > 0

    # Read from file
    loaded_dataset = read_dataset_from_file(TabularDataset, str(file_path))

    # Check that loaded dataset matches original
    assert isinstance(loaded_dataset, TabularDataset)
    assert len(loaded_dataset) == len(dataset)
    pd.testing.assert_frame_equal(loaded_dataset.to_pandas(), dataset.to_pandas())


def test_file_storage_error_handling(tmp_path):
    """Test error handling for file storage operations."""
    # Test with non-existent file
    non_existent_path = tmp_path / "non_existent.bin"
    with pytest.raises(FileNotFoundError):
        read_dataset_from_file(TabularDataset, str(non_existent_path))

    # Test with invalid file content
    invalid_path = tmp_path / "invalid.bin"
    with open(str(invalid_path), "wb") as f:
        f.write(b"invalid data")

    with pytest.raises(RuntimeError):
        read_dataset_from_file(TabularDataset, str(invalid_path))

    # Test with invalid directory path
    invalid_dir = tmp_path / "non_existent_dir" / "dataset.bin"

    # Create a dataset
    df = pd.DataFrame({"a": [1, 2, 3]})
    dataset = TabularDataset(df)

    with pytest.raises(FileNotFoundError):
        write_dataset_to_file(dataset, str(invalid_dir))


@pytest.mark.skipif(
    not hasattr(__import__("multiprocessing", fromlist=["shared_memory"]), "shared_memory"),
    reason="Shared memory requires Python 3.8+",
)
def test_shared_memory_error_handling():
    """Test error handling in shared memory functions."""
    # Mock an ImportError for dataset_to_shared_memory
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "multiprocessing" or name == "multiprocessing.shared_memory":
            raise ImportError("Mocked import error")
        return original_import(name, *args, **kwargs)

    # Test with ImportError
    try:
        builtins.__import__ = mock_import

        # Test dataset_to_shared_memory
        dataset = TabularDataset(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(ImportError):
            dataset_to_shared_memory(dataset, "test_segment")

        # Test dataset_from_shared_memory
        with pytest.raises(ImportError):
            dataset_from_shared_memory(TabularDataset, "test_segment")
    finally:
        builtins.__import__ = original_import

    # Restore original import function
    builtins.__import__ = original_import

    # Test with non-existent shared memory segment
    try:
        with pytest.raises((FileNotFoundError, ValueError)):
            dataset_from_shared_memory(TabularDataset, "non_existent_segment_name")
    except ImportError:
        pytest.skip("shared_memory not available")
