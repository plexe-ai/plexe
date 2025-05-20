"""
Tools for dataset manipulation, splitting, and registration.

These tools help with dataset operations within the model generation pipeline, including
splitting datasets into training, validation, and test sets, registering datasets with
the dataset registry, creating sample data for validation, previewing dataset content,
and registering exploratory data analysis (EDA) reports.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from smolagents import tool

from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.registries.objects import ObjectRegistry

logger = logging.getLogger(__name__)


@tool
def register_split_datasets(
    dataset_names: List[str],
    train_datasets: List[pd.DataFrame],
    validation_datasets: List[pd.DataFrame],
    test_datasets: List[pd.DataFrame],
) -> Dict[str, List[str]]:
    """
    Register train, validation, and test datasets in the object registry after custom splitting.

    This tool allows the agent to register datasets after performing custom splitting logic.

    Args:
        dataset_names: Original dataset names that were split
        train_datasets: List of pandas DataFrames containing training data
        validation_datasets: List of pandas DataFrames containing validation data
        test_datasets: List of pandas DataFrames containing test data

    Returns:
        Dictionary containing lists of registered dataset names:
        {
            "train_datasets": List of training dataset names,
            "validation_datasets": List of validation dataset names,
            "test_datasets": List of test dataset names,
            "dataset_sizes": Dictionary with sizes of each dataset type
        }
    """
    if (
        len(dataset_names) != len(train_datasets)
        or len(dataset_names) != len(validation_datasets)
        or len(dataset_names) != len(test_datasets)
    ):
        raise ValueError("The number of dataset names must match the number of train, validation, and test datasets")

    # Initialize the dataset registry
    object_registry = ObjectRegistry()

    # Initialize dataset name lists
    train_dataset_names = []
    validation_dataset_names = []
    test_dataset_names = []

    # Initialize the dataset sizes dictionary
    dataset_sizes = {"train": [], "validation": [], "test": []}

    # Register each split dataset
    for i, name in enumerate(dataset_names):
        # Convert pandas DataFrames to TabularDataset objects
        train_ds = DatasetAdapter.coerce(train_datasets[i])
        val_ds = DatasetAdapter.coerce(validation_datasets[i])
        test_ds = DatasetAdapter.coerce(test_datasets[i])

        # Register split datasets in the registry
        train_name = f"{name}_train"
        val_name = f"{name}_val"
        test_name = f"{name}_test"

        object_registry.register(TabularConvertible, train_name, train_ds, overwrite=True)
        object_registry.register(TabularConvertible, val_name, val_ds, overwrite=True)
        object_registry.register(TabularConvertible, test_name, test_ds, overwrite=True)

        # Store dataset names
        train_dataset_names.append(train_name)
        validation_dataset_names.append(val_name)
        test_dataset_names.append(test_name)

        # Store dataset sizes
        dataset_sizes["train"].append(len(train_ds))
        dataset_sizes["validation"].append(len(val_ds))
        dataset_sizes["test"].append(len(test_ds))

        logger.debug(
            f"✅ Registered custom split of dataset {name} into train/validation/test with sizes "
            f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
        )

    return {
        "train_datasets": train_dataset_names,
        "validation_datasets": validation_dataset_names,
        "test_datasets": test_dataset_names,
        "dataset_sizes": dataset_sizes,
    }


@tool
def create_input_sample(input_schema: Dict[str, str], n_samples: int = 5) -> bool:
    """
    Create and register a synthetic sample input dataset that matches the model's input schema.
    This sample is used for validating inference code.

    Args:
        input_schema: Dictionary mapping field names to their types
        n_samples: Number of samples to generate (default: 5)

    Returns:
        True if sample was successfully created and registered, False otherwise
    """
    object_registry = ObjectRegistry()

    try:
        # Create synthetic sample data that matches the schema
        input_sample_dicts = []

        # Generate synthetic examples
        for i in range(n_samples):
            sample = {}
            for field_name, field_type in input_schema.items():
                # Generate appropriate sample values based on type
                if field_type == "int":
                    sample[field_name] = i * 10
                elif field_type == "float":
                    sample[field_name] = i * 10.5
                elif field_type == "bool":
                    sample[field_name] = i % 2 == 0
                elif field_type == "str":
                    sample[field_name] = f"sample_{field_name}_{i}"
                else:
                    sample[field_name] = None
            input_sample_dicts.append(sample)

        # TODO: we should use an LLM call to generate sensible values; then validate using pydantic

        # Register the input sample in the registry for validation tool to use
        object_registry.register(list, "predictor_input_sample", input_sample_dicts)
        logger.debug(
            f"✅ Registered synthetic input sample with {len(input_sample_dicts)} examples for inference validation"
        )
        return True

    except Exception as e:
        logger.warning(f"⚠️ Error creating input sample for validation: {str(e)}")
        return False


@tool
def drop_null_columns(dataset_name: str) -> Dict[str, str]:
    """
    Drop all columns from the dataset that are completely null and register the modified dataset.

    Args:
        dataset_name: Name of the dataset to modify

    Returns:
        Dictionary containing results of the operation:
        - dataset_name: Name of the modified dataset
        - n_dropped: Number of columns dropped
    """
    object_registry = ObjectRegistry()

    try:
        # Get dataset from registry
        dataset = object_registry.get(TabularConvertible, dataset_name)
        df = dataset.to_pandas()

        # Drop columns with all null values TODO: make this more intelligent
        # Drop columns with >=50% missing values
        null_columns = df.columns[df.isnull().mean() >= 0.5]

        # Drop constant columns (zero variance)
        constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

        # Drop quasi-constant columns (e.g., one value in >95% of rows)
        quasi_constant_columns = [
            col for col in df.columns if (df[col].value_counts(dropna=False, normalize=True).values[0] > 0.95)
        ]

        # Drop columns with all unique values (likely IDs)
        unique_columns = [col for col in df.columns if df[col].nunique(dropna=False) == len(df)]

        # Drop duplicate columns
        duplicate_columns = []
        seen = {}
        for col in df.columns:
            col_data = df[col].to_numpy()
            key = col_data.tobytes() if hasattr(col_data, "tobytes") else tuple(col_data)
            if key in seen:
                duplicate_columns.append(col)
            else:
                seen[key] = col

        # Combine all columns to drop (set to avoid duplicates)
        all_bad_columns = (
            set(null_columns)
            | set(constant_columns)
            | set(quasi_constant_columns)
            | set(unique_columns)
            | set(duplicate_columns)
        )
        n_dropped = len(all_bad_columns)
        df.drop(columns=list(all_bad_columns), inplace=True)

        # Unregister the original dataset
        object_registry.delete(TabularConvertible, dataset_name)

        # Register the modified dataset
        object_registry.register(TabularConvertible, dataset_name, DatasetAdapter.coerce(df))

        logger.debug(f"✅ Dropped {n_dropped} null columns from dataset '{dataset_name}'")
        return {
            "dataset_name": dataset_name,
            "n_dropped": n_dropped,
        }

    except Exception as e:
        logger.warning(f"⚠️ Error dropping null columns: {str(e)}")
        return {
            "error": f"Failed to drop null columns from dataset '{dataset_name}': {str(e)}",
            "dataset_name": dataset_name,
            "n_dropped": 0,
        }


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


@tool
def register_eda_report(
    dataset_name: str,
    overview: Dict[str, Any],
    feature_engineering_opportunities: Dict[str, Any],
    data_quality_challenges: Dict[str, Any],
    data_preprocessing_requirements: Dict[str, Any],
    feature_importance: Dict[str, Any],
    insights: List[str],
    recommendations: List[str],
) -> bool:
    """
    Register an exploratory data analysis (EDA) report for a dataset in the Object Registry.

    This tool creates a structured report with actionable ML engineering insights from exploratory
    data analysis and registers it in the Object Registry for use by other agents.

    Args:
        dataset_name: Name of the dataset that was analyzed
        overview: Essential dataset statistics including target variable analysis
        feature_engineering_opportunities: Specific transformation needs, interaction effects,
                                          and engineered features that would improve model performance
        data_quality_challenges: Critical data issues with specific handling recommendations
        data_preprocessing_requirements: Necessary preprocessing steps with clear justification
        feature_importance: Assessment of feature predictive potential and relevance
        insights: Key insights derived from the analysis that directly impact feature engineering
        recommendations: Specific, prioritized actions for preprocessing and feature engineering

    Returns:
        True if the report was successfully registered, False otherwise
    """
    object_registry = ObjectRegistry()

    try:
        # Create structured EDA report with actionable ML focus
        eda_report = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overview": overview,
            "feature_engineering_opportunities": feature_engineering_opportunities,
            "data_quality_challenges": data_quality_challenges,
            "data_preprocessing_requirements": data_preprocessing_requirements,
            "feature_importance": feature_importance,
            "insights": insights,
            "recommendations": recommendations,
        }

        # Register in registry
        object_registry.register(dict, f"eda_report_{dataset_name}", eda_report)
        logger.debug(f"✅ Registered EDA report for dataset '{dataset_name}'")
        return True

    except Exception as e:
        logger.warning(f"⚠️ Error registering EDA report: {str(e)}")
        return False


@tool
def get_eda_report(dataset_name: str) -> Dict[str, Any]:
    """
    Retrieve an exploratory data analysis (EDA) report for a dataset generated by a data analyst.

    This tool fetches the EDA report previously created by the data analysis agent, containing
    comprehensive findings about the dataset's structure, features, relationships, and quality.

    Args:
        dataset_name: Name of the dataset to retrieve the EDA report for

    Returns:
        Dictionary containing the complete EDA report
    """
    object_registry = ObjectRegistry()

    try:
        # If name ends in _train, _val, or _test, strip it to get the original dataset name
        if dataset_name.endswith(("_train", "_val", "_test")):
            dataset_name = dataset_name.rsplit("_", 1)[0]

        # Check if EDA report exists
        report_key = f"eda_report_{dataset_name}"

        # Get the report from registry
        eda_report = object_registry.get(dict, report_key)
        logger.debug(f"✅ Retrieved EDA report for dataset '{dataset_name}'")
        return eda_report

    except KeyError:
        # Report not found
        logger.warning(f"⚠️ No EDA report found for dataset '{dataset_name}'")
        return {
            "error": f"No EDA report found for dataset '{dataset_name}'",
            "dataset_name": dataset_name,
            "available": False,
        }

    except Exception as e:
        logger.warning(f"⚠️ Error retrieving EDA report: {str(e)}")
        return {
            "error": f"Failed to retrieve EDA report for dataset '{dataset_name}': {str(e)}",
            "dataset_name": dataset_name,
            "available": False,
        }
