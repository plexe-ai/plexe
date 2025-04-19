"""
Tests for the dataset interface.

This module verifies:
1. Creation and usage of DatasetStructure
2. Basic attributes and operations of DatasetStructure
3. Abstract methods in Dataset must be implemented by subclasses
4. Error handling when trying to instantiate Dataset directly
"""

import pytest
from plexe.internal.common.datasets.interface import Dataset, DatasetStructure


def test_dataset_structure_creation():
    """Test creation of a DatasetStructure with valid parameters."""
    # Create a simple structure
    structure = DatasetStructure(
        modality="table", features=["feature1", "feature2", "target"], details={"num_rows": 10, "num_columns": 3}
    )

    # Check attributes
    assert structure.modality == "table"
    assert structure.features == ["feature1", "feature2", "target"]
    assert structure.details["num_rows"] == 10
    assert structure.details["num_columns"] == 3


def test_dataset_structure_tensor_modality():
    """Test creation of a DatasetStructure with tensor modality."""
    # Create a tensor structure
    structure = DatasetStructure(
        modality="tensor", features=["pixel_values"], details={"shape": [32, 32, 3], "dtype": "float32"}
    )

    # Check attributes
    assert structure.modality == "tensor"
    assert structure.features == ["pixel_values"]
    assert structure.details["shape"] == [32, 32, 3]
    assert structure.details["dtype"] == "float32"


def test_dataset_structure_other_modality():
    """Test creation of a DatasetStructure with 'other' modality."""
    # Create a structure with 'other' modality
    structure = DatasetStructure(
        modality="other", features=["custom_data"], details={"type": "custom", "format": "specialized"}
    )

    # Check attributes
    assert structure.modality == "other"
    assert structure.features == ["custom_data"]
    assert structure.details["type"] == "custom"
    assert structure.details["format"] == "specialized"


# Note: Python doesn't enforce Literal type annotations at runtime,
# so we're not testing invalid modality values since they would pass
# in a regular Python environment. In a strictly typed environment
# or with runtime type checking, this would raise an error.


class MinimalDataset(Dataset):
    """Minimal implementation of Dataset for testing."""

    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify_column=None, random_state=None):
        return self, self, self

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        return self

    def to_bytes(self):
        return b"minimal_dataset"

    @classmethod
    def from_bytes(cls, data):
        return cls()

    @property
    def structure(self):
        return DatasetStructure(modality="other", features=["dummy"], details={})

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None


class IncompleteDataset(Dataset):
    """Dataset implementation missing required methods."""

    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify_column=None, random_state=None):
        return self, self, self

    # Missing other required methods


def test_dataset_instantiation():
    """Test that Dataset can't be instantiated directly."""
    with pytest.raises(TypeError):
        Dataset()  # Should raise TypeError: Can't instantiate abstract class


def test_minimal_dataset():
    """Test that a minimal implementation of Dataset can be instantiated."""
    dataset = MinimalDataset()
    assert isinstance(dataset, Dataset)

    # Test basic functionality
    train, val, test = dataset.split()
    assert isinstance(train, MinimalDataset)
    assert isinstance(val, MinimalDataset)
    assert isinstance(test, MinimalDataset)

    sample = dataset.sample(n=5)
    assert isinstance(sample, MinimalDataset)

    data_bytes = dataset.to_bytes()
    assert data_bytes == b"minimal_dataset"

    restored = MinimalDataset.from_bytes(data_bytes)
    assert isinstance(restored, MinimalDataset)

    assert len(dataset) == 0
    assert dataset[0] is None

    structure = dataset.structure
    assert structure.modality == "other"
    assert structure.features == ["dummy"]


def test_incomplete_dataset():
    """Test that a Dataset implementation missing required methods raises errors."""
    with pytest.raises(TypeError):
        IncompleteDataset()  # Should raise TypeError for missing abstract methods
