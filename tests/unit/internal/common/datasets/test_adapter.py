"""
Tests for the DatasetAdapter class.

This module verifies:
1. Conversion of different data formats to appropriate dataset types
2. Auto-detection of dataset types
3. Feature extraction functionality
4. Error handling for unsupported dataset types
"""

import pytest
import pandas as pd

from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import Dataset, DatasetStructure
from plexe.internal.common.datasets.tabular import TabularDataset


class MockDataset(Dataset):
    """Mock dataset implementation for testing."""

    def __init__(self, features=None):
        self._features = features or ["feature1", "feature2"]

    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify_column=None, random_state=None):
        return self, self, self

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        return self

    def to_bytes(self):
        return b"mock_dataset"

    @classmethod
    def from_bytes(cls, data):
        return cls()

    @property
    def structure(self):
        return DatasetStructure(modality="other", features=self._features, details={"mock": True})

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"item": idx}


def test_adapter_coerce_pandas():
    """Test that DatasetAdapter.coerce handles pandas DataFrames."""
    # Create test data
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [0, 1, 0]})

    # Coerce to dataset
    result = DatasetAdapter.coerce(df)

    # Check that the result is a TabularDataset
    assert isinstance(result, TabularDataset)
    assert len(result) == 3

    # Check that the data was preserved
    pd.testing.assert_frame_equal(result.to_pandas(), df)


def test_adapter_coerce_dataset():
    """Test that DatasetAdapter.coerce passes through Dataset instances."""
    # Create test data
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [0, 1, 0]})

    # Create TabularDataset
    dataset = TabularDataset(df)

    # Coerce the dataset
    result = DatasetAdapter.coerce(dataset)

    # Check that the result is the same TabularDataset
    assert result is dataset


def test_adapter_auto_detect():
    """Test the auto_detect functionality."""
    # Test with pandas DataFrame
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert DatasetAdapter.auto_detect(df) == "tabular"

    # Test with unsupported type
    assert DatasetAdapter.auto_detect("not a dataset") is None


def test_adapter_coerce_unsupported():
    """Test error handling for unsupported dataset types."""
    # Try to coerce an unsupported type
    with pytest.raises(ValueError):
        DatasetAdapter.coerce("not a dataset")


def test_adapter_features():
    """Test the features extraction functionality."""
    # Create test datasets
    df1 = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})

    df2 = pd.DataFrame({"feature3": [4, 5, 6], "feature4": ["d", "e", "f"]})

    dataset1 = TabularDataset(df1)
    dataset2 = TabularDataset(df2)

    # Create dataset dictionary
    datasets = {"dataset1": dataset1, "dataset2": dataset2}

    # Extract features
    features = DatasetAdapter.features(datasets)

    # Check that features were correctly extracted
    assert len(features) == 4
    assert "dataset1.feature1" in features
    assert "dataset1.feature2" in features
    assert "dataset2.feature3" in features
    assert "dataset2.feature4" in features
