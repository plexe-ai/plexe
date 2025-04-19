"""
This module provides the DatasetAdapter class, which converts various dataset formats into standardized Dataset
objects. This enables the library to accept multiple dataset types as inputs, while ensuring consistency and
interoperability.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from plexe.internal.common.datasets.interface import Dataset
from plexe.internal.common.datasets.tabular import TabularDataset


logger = logging.getLogger(__name__)


# This dictionary defines the mapping of dataset types to their respective wrapper classes.
# Note: this could be replaced with a separate DatasetRegistry class in the future if we need dynamic registration
# TODO: Add more dataset types and their corresponding classes
DATASET_REGISTRY_MAP = {
    "tabular": TabularDataset,
}


class DatasetAdapter:
    """
    A utility class for converting different dataset formats into standardized Dataset objects.

    This class provides methods for handling structured datasets, ensuring compatibility with downstream
    processing steps in the plexe library.
    """

    @staticmethod
    def coerce(dataset: Any) -> Dataset:
        """
        Converts a dataset to a standardized format.

        This method attempts to convert the input dataset to a Dataset implementation if a suitable one is
        available. If dataset_type is None, it tries to auto-detect the appropriate type. For backward compatibility,
        it falls back to returning a pandas DataFrame if no appropriate Dataset is available.

        :param dataset: The dataset to convert
        :returns: A Dataset implementation or pandas DataFrame
        :raises ValueError: If the dataset type is unsupported
        """
        # If dataset is already a Dataset, return it directly
        if isinstance(dataset, Dataset):
            return dataset

        # Determine the dataset type
        dataset_type = DatasetAdapter.auto_detect(dataset)

        # If we have a suitable Dataset implementation, use it
        if dataset_type is not None:
            try:
                return DATASET_REGISTRY_MAP[dataset_type](dataset)
            except (ValueError, ImportError) as e:
                # Log the error but continue with the fallback
                logger.error(f"Failed to create {dataset_type} dataset: {str(e)}")
                raise ValueError(f"Failed to convert dataset of type {type(dataset)} to {dataset_type}: {str(e)}")
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    @classmethod
    def auto_detect(cls, data: Any) -> Optional[str]:
        """
        Auto-detect the appropriate dataset type for the given data.

        :param data: The data to detect the appropriate dataset type for
        :returns: Name of the detected dataset implementation, or None if no appropriate type was found
        """
        if isinstance(data, pd.DataFrame):
            return "tabular"

        # TODO: Add more auto-detection logic for other data types
        return None

    @staticmethod
    def features(datasets: Dict[str, Dataset]) -> List[str]:
        """
        Extracts a flat list of feature names from the given datasets.

        This method is useful for gathering meaningful names for all features available across multiple datasets,
        which can be passed to a downstream LLM call or other processing steps.

        :param datasets: A dictionary of dataset names and their corresponding datasets
        :returns: A list of feature names
        :raises ValueError: If the dataset type is unsupported
        """
        features = []
        for name, dataset in datasets.items():
            features.extend(f"{name}.{feature}" for feature in dataset.structure.features)
        return features
