"""
Tools for dataset manipulation, splitting, and registration.

These tools help with dataset operations within the model generation pipeline, including
splitting datasets into training, validation, and test sets, registering datasets with
the dataset registry, creating sample data for validation, and previewing dataset content.
"""

import logging
from typing import Dict, List, Any

import numpy as np
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

    logger.debug("🔪 Splitting datasets into train, validation, and test sets")
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

        logger.debug(
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

        # Keep only columns that match the input schema and convert to list of dicts
        input_sample_df = combined_df[input_schema_fields].head(min(100, len(combined_df)))
        input_sample_dicts = input_sample_df.to_dict(orient="records")

        # Register the input sample in the registry for validation tool to use
        object_registry.register(list, "predictor_input_sample", input_sample_dicts)
        logger.debug(f"✅ Registered input sample with {len(input_sample_dicts)} dictionaries for inference validation")
        return True

    except Exception as e:
        logger.warning(f"⚠️ Error creating input sample for validation: {str(e)}")
        return False


@tool
def get_dataset_preview(dataset_name: str) -> Dict[str, Any]:
    """
    Generate a concise preview of a dataset with statistical information to help agents understand the data.

    Args:
        dataset_name: Name of the dataset to preview

    Returns:
        Dictionary containing dataset information:
        - shape: dimensions of the dataset
        - dtypes: data types of columns
        - summary_stats: basic statistics (mean, median, min/max)
        - missing_values: count of missing values per column
        - sample_rows: sample of the data (5 rows)
    """
    object_registry = ObjectRegistry()

    try:
        # Get dataset from registry
        dataset = object_registry.get(TabularConvertible, dataset_name)
        df = dataset.to_pandas()

        # Basic shape and data types
        result = {
            "dataset_name": dataset_name,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict(orient="records"),
        }

        # Basic statistics
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().to_dict()
            result["summary_stats"] = {
                col: {
                    "mean": stats[col].get("mean"),
                    "std": stats[col].get("std"),
                    "min": stats[col].get("min"),
                    "25%": stats[col].get("25%"),
                    "median": stats[col].get("50%"),
                    "75%": stats[col].get("75%"),
                    "max": stats[col].get("max"),
                }
                for col in numeric_cols
            }

        # Missing values
        missing_counts = df.isnull().sum().to_dict()
        result["missing_values"] = {col: count for col, count in missing_counts.items() if count > 0}

        return result

    except Exception as e:
        logger.warning(f"⚠️ Error creating dataset preview: {str(e)}")
        return {
            "error": f"Failed to generate preview for dataset '{dataset_name}': {str(e)}",
            "dataset_name": dataset_name,
        }
