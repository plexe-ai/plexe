"""
Tools for dataset manipulation, splitting, and registration.

These tools help with dataset operations within the model generation pipeline, including
splitting datasets into training, validation, and test sets, registering datasets with
the dataset registry, and creating sample data for validation.
"""

import logging
from typing import Dict, List
import pandas as pd
from smolagents import tool

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.registries.objects import ObjectRegistry

logger = logging.getLogger(__name__)


@tool
def split_datasets(
    datasets: List[str],
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
) -> Dict[str, List[str]]:
    """
    Split datasets into train, validation, and test sets and register the new split datasets with
    the dataset registry. After splitting and registration, the new dataset names can be used as valid references
    for datasets.

    Args:
        datasets: List of names for the datasets that need to be split
        train_ratio: Ratio of data to use for training (default: 0.9)
        val_ratio: Ratio of data to use for validation (default: 0.1)
        test_ratio: Ratio of data to use for testing (default: 0.0)

    Returns:
        Dictionary containing lists of registered dataset names:
        {
            "train_datasets": List of training dataset names,
            "validation_datasets": List of validation dataset names,
            "test_datasets": List of test dataset names
        }
    """
    # Initialize the dataset registry
    object_registry = ObjectRegistry()

    # Initialize dataset name lists
    train_dataset_names = []
    validation_dataset_names = []
    test_dataset_names = []

    logger.info("🔪 Splitting datasets into train, validation, and test sets")
    for name in datasets:
        dataset = object_registry.get(TabularConvertible, name)
        train_ds, val_ds, test_ds = dataset.split(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

        # Register split datasets in the registry
        train_name = f"{name}_train"
        val_name = f"{name}_val"
        test_name = f"{name}_test"

        object_registry.register(TabularConvertible, train_name, train_ds)
        object_registry.register(TabularConvertible, val_name, val_ds)
        object_registry.register(TabularConvertible, test_name, test_ds)

        # Store dataset names
        train_dataset_names.append(train_name)
        validation_dataset_names.append(val_name)
        test_dataset_names.append(test_name)

        logger.info(
            f"✅ Split dataset {name} into train/validation/test with sizes "
            f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
        )

    return {
        "train_datasets": train_dataset_names,
        "validation_datasets": validation_dataset_names,
        "test_datasets": test_dataset_names,
    }


@tool
def create_input_sample(train_dataset_names: List[str], input_schema_fields: List[str]) -> bool:
    """
    Create and register a sample input dataset for inference code validation.

    Args:
        train_dataset_names: List of training dataset names to extract samples from
        input_schema_fields: List of field names from the input schema

    Returns:
        True if sample was successfully created and registered, False otherwise
    """
    object_registry = ObjectRegistry()

    try:
        # Concatenate all train datasets and extract relevant columns for the input schema
        input_sample_dfs = []
        for dataset_name in train_dataset_names:
            dataset = object_registry.get(TabularConvertible, dataset_name)
            df = dataset.to_pandas().head(max(5, 5 // len(train_dataset_names)))
            input_sample_dfs.append(df)

        if not input_sample_dfs:
            logger.warning("⚠️ No datasets available to create input sample for validation")
            return False

        # Combine datasets and filter for input schema columns
        combined_df = pd.concat(input_sample_dfs, axis=0).reset_index(drop=True)

        # Keep only columns that match the input schema
        input_sample_df = combined_df[input_schema_fields].head(5)

        # Register the input sample in the registry for validation tool to use
        object_registry.register(pd.DataFrame, "predictor_input_sample", input_sample_df)
        logger.info(f"✅ Registered input sample with {len(input_sample_df)} rows for inference validation")
        return True

    except Exception as e:
        logger.warning(f"⚠️ Error creating input sample for validation: {str(e)}")
        return False
