"""
Tests for the TabularDataset implementation.

This module verifies:
1. Creation of TabularDatasets from pandas DataFrames
2. Dataset splitting into train/val/test sets
3. Dataset sampling functionality
4. Serialization and deserialization
5. Dataset structure metadata
6. Conversion to pandas and numpy formats
7. Basic operations like __len__ and __getitem__
"""

import pytest
import pandas as pd
import numpy as np

from plexe.internal.common.datasets.tabular import TabularDataset
from plexe.internal.common.datasets.interface import DatasetStructure
from plexe.internal.common.utils.dataset_storage import write_dataset_to_file, read_dataset_from_file


def test_tabular_dataset_creation():
    """Test that TabularDataset can be created from pandas DataFrame."""
    # Create test data
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": ["a", "b", "c", "d", "e"], "target": [0, 1, 0, 1, 0]})

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Check that the dataset was created correctly
    assert len(dataset) == 5
    assert isinstance(dataset.to_pandas(), pd.DataFrame)
    assert isinstance(dataset.to_numpy(), np.ndarray)
    pd.testing.assert_frame_equal(dataset.to_pandas(), df)


def test_tabular_dataset_validation():
    """Test validation of input data types."""
    # Valid input
    df = pd.DataFrame({"a": [1, 2, 3]})
    TabularDataset(df)  # Should work

    # Invalid input
    with pytest.raises(ValueError):
        TabularDataset("not a dataframe")

    with pytest.raises(ValueError):
        TabularDataset([1, 2, 3])


def test_tabular_dataset_split_standard():
    """Test standard train/val/test split with default ratios."""
    # Create test data
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": [f"val_{i}" for i in range(100)], "target": [i % 2 for i in range(100)]}
    )

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Split dataset with default ratios (0.7, 0.15, 0.15)
    train, val, test = dataset.split()

    # Check that each split is a TabularDataset
    assert isinstance(train, TabularDataset)
    assert isinstance(val, TabularDataset)
    assert isinstance(test, TabularDataset)

    # Check split sizes
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Check total size
    assert len(train) + len(val) + len(test) == 100


def test_tabular_dataset_split_custom_ratios():
    """Test train/val/test split with custom ratios."""
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": [f"val_{i}" for i in range(100)], "target": [i % 2 for i in range(100)]}
    )

    dataset = TabularDataset(df)

    # Split with custom ratios
    train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Check split sizes
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Check total size
    assert len(train) + len(val) + len(test) == 100


def test_tabular_dataset_split_stratified():
    """Test stratified splitting."""
    df = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": [f"val_{i}" for i in range(100)],
            "target": [i % 2 for i in range(100)],  # 50/50 split of 0s and 1s
        }
    )

    dataset = TabularDataset(df)

    # Split with stratification
    train, val, test = dataset.split(stratify_column="target")

    # Check that class proportions are maintained
    assert abs(train.to_pandas()["target"].mean() - 0.5) < 0.1
    assert abs(val.to_pandas()["target"].mean() - 0.5) < 0.1
    assert abs(test.to_pandas()["target"].mean() - 0.5) < 0.1


def test_tabular_dataset_split_reproducibility():
    """Test that splits are reproducible with same random state."""
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": [f"val_{i}" for i in range(100)], "target": [i % 2 for i in range(100)]}
    )

    dataset = TabularDataset(df)

    # Create two splits with the same random state
    train1, val1, test1 = dataset.split(random_state=42)
    train2, val2, test2 = dataset.split(random_state=42)

    # Check that the splits are identical
    pd.testing.assert_frame_equal(train1.to_pandas(), train2.to_pandas())
    pd.testing.assert_frame_equal(val1.to_pandas(), val2.to_pandas())
    pd.testing.assert_frame_equal(test1.to_pandas(), test2.to_pandas())


def test_tabular_dataset_split_edge_cases():
    """Test edge cases for splitting."""
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": [f"val_{i}" for i in range(100)], "target": [i % 2 for i in range(100)]}
    )

    dataset = TabularDataset(df)

    # All data to train
    train, val, test = dataset.split(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    assert len(train) == 100
    assert len(val) == 0
    assert len(test) == 0

    # No validation set
    train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)
    assert len(train) == 80
    assert len(val) == 0
    assert len(test) == 20

    # No test set
    train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    assert len(train) == 80
    assert len(val) == 20
    assert len(test) == 0

    # Invalid ratios
    with pytest.raises(ValueError):
        dataset.split(train_ratio=0.8, val_ratio=0.3, test_ratio=0.2)  # Sum > 1


def test_tabular_dataset_sample():
    """Test sampling functionality."""
    # Create test data
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": [f"val_{i}" for i in range(100)], "target": [i % 2 for i in range(100)]}
    )

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Sample by count
    sampled = dataset.sample(n=10, random_state=42)
    assert len(sampled) == 10
    assert isinstance(sampled, TabularDataset)

    # Sample by fraction
    sampled = dataset.sample(frac=0.1, random_state=42)
    assert len(sampled) == 10
    assert isinstance(sampled, TabularDataset)

    # Sample with replacement
    sampled = dataset.sample(n=120, replace=True, random_state=42)
    assert len(sampled) == 120
    assert isinstance(sampled, TabularDataset)

    # Check reproducibility
    sampled1 = dataset.sample(n=10, random_state=42)
    sampled2 = dataset.sample(n=10, random_state=42)
    pd.testing.assert_frame_equal(sampled1.to_pandas(), sampled2.to_pandas())


def test_tabular_dataset_serialization():
    """Test that TabularDataset can be serialized and deserialized."""
    # Create test data
    df = pd.DataFrame(
        {"feature1": range(10), "feature2": [f"val_{i}" for i in range(10)], "target": [i % 2 for i in range(10)]}
    )

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Serialize to bytes
    data_bytes = dataset.to_bytes()
    assert isinstance(data_bytes, bytes)
    assert len(data_bytes) > 0

    # Deserialize from bytes
    deserialized = TabularDataset.from_bytes(data_bytes)

    # Check that the deserialized dataset matches the original
    assert len(deserialized) == len(dataset)
    pd.testing.assert_frame_equal(deserialized.to_pandas(), dataset.to_pandas())


def test_tabular_dataset_serialization_error_handling():
    """Test error handling during serialization/deserialization."""
    # Create a TabularDataset
    dataset = TabularDataset(pd.DataFrame({"a": [1, 2, 3]}))

    # Test serialization (should succeed)
    dataset.to_bytes()

    # Test deserialization with invalid data
    with pytest.raises(RuntimeError):
        TabularDataset.from_bytes(b"invalid data")


def test_tabular_dataset_structure():
    """Test structure property."""
    # Create test data
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": ["a", "b", "c", "d", "e"], "target": [0, 1, 0, 1, 0]})

    # Create TabularDataset
    dataset = TabularDataset(df)
    structure = dataset.structure

    # Check structure type and fields
    assert isinstance(structure, DatasetStructure)
    assert structure.modality == "table"
    assert set(structure.features) == {"feature1", "feature2", "target"}

    # Check details
    assert structure.details["num_rows"] == 5
    assert structure.details["num_columns"] == 3
    assert set(structure.details["column_names"]) == {"feature1", "feature2", "target"}
    assert isinstance(structure.details["column_types"], dict)


def test_tabular_dataset_file_storage(tmp_path):
    """Test that TabularDataset can be stored to and loaded from a file."""
    # Create test data
    df = pd.DataFrame(
        {"feature1": range(10), "feature2": [f"val_{i}" for i in range(10)], "target": [i % 2 for i in range(10)]}
    )

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Write to file
    file_path = tmp_path / "test_dataset.bin"
    write_dataset_to_file(dataset, str(file_path))

    # Check that the file exists
    assert file_path.exists()

    # Read from file
    loaded = read_dataset_from_file(TabularDataset, str(file_path))

    # Check that the loaded dataset matches the original
    assert len(loaded) == len(dataset)
    pd.testing.assert_frame_equal(loaded.to_pandas(), dataset.to_pandas())


def test_tabular_dataset_conversion():
    """Test conversion to pandas and numpy."""
    # Create test data
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": ["a", "b", "c", "d", "e"], "target": [0, 1, 0, 1, 0]})

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Test pandas conversion
    out_df = dataset.to_pandas()
    assert isinstance(out_df, pd.DataFrame)
    pd.testing.assert_frame_equal(out_df, df)

    # Test numpy conversion
    arr = dataset.to_numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, df.to_numpy())


def test_tabular_dataset_getitem():
    """Test __getitem__ functionality."""
    # Create test data
    df = pd.DataFrame(
        {"feature1": range(10), "feature2": [f"val_{i}" for i in range(10)], "target": [i % 2 for i in range(10)]}
    )

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Single item access
    item = dataset[0]
    assert item.feature1 == 0
    assert item.feature2 == "val_0"
    assert item.target == 0

    # Slice access
    slice_items = dataset[1:4]
    assert isinstance(slice_items, pd.Series) or isinstance(slice_items, pd.DataFrame)

    if isinstance(slice_items, pd.DataFrame):
        assert len(slice_items) == 3
    else:
        # If it returns a Series, check that the values match
        expected_series = df.iloc[1:4]
        if isinstance(expected_series, pd.Series):
            pd.testing.assert_series_equal(slice_items, expected_series)


def test_tabular_dataset_len():
    """Test __len__ functionality."""
    # Create datasets of different sizes
    dataset1 = TabularDataset(pd.DataFrame({"a": range(5)}))
    dataset2 = TabularDataset(pd.DataFrame({"a": range(10)}))

    # Check lengths
    assert len(dataset1) == 5
    assert len(dataset2) == 10
