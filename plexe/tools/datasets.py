"""
Tools for dataset manipulation, splitting, and registration.

These tools help with dataset operations within the model generation pipeline, including
splitting datasets into training, validation, and test sets, registering datasets with
the dataset registry, creating sample data for validation, previewing dataset content,
registering exploratory data analysis (EDA) reports, and registering feature engineering results.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from smolagents import tool

from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code

logger = logging.getLogger(__name__)


@tool
def register_split_datasets(
    dataset_name: str,
    train_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    splitting_code: str,
) -> Dict[str, str]:
    """
    Register train, validation, and test datasets in the object registry after custom splitting.
    This tool allows the agent to register datasets after performing custom splitting logic.

    Args:
        dataset_name: Original name of the dataset that was split
        train_dataset: pandas DataFrame containing training data
        validation_dataset: pandas DataFrame containing validation data
        test_dataset: pandas DataFrame containing test data
        splitting_code: the code that was used to split the dataset

    Returns:
        Dictionary containing lists of registered dataset names:
        {
            "train_dataset": name of the training dataset,
            "validation_dataset": name of the validation dataset,
            "test_dataset": name of the test dataset,
            "dataset_size": Dictionary with sizes of each dataset
        }
    """

    # Initialize the dataset registry
    object_registry = ObjectRegistry()

    # Initialize the dataset sizes dictionary
    dataset_sizes = {"train": [], "validation": [], "test": []}

    # Register each split dataset
    # Convert pandas DataFrames to TabularDataset objects
    train_ds = DatasetAdapter.coerce(train_dataset)
    val_ds = DatasetAdapter.coerce(validation_dataset)
    test_ds = DatasetAdapter.coerce(test_dataset)

    # Register split datasets in the registry
    train_name = f"{dataset_name}_train"
    val_name = f"{dataset_name}_val"
    test_name = f"{dataset_name}_test"

    object_registry.register(TabularConvertible, train_name, train_ds, overwrite=True, immutable=True)
    object_registry.register(TabularConvertible, val_name, val_ds, overwrite=True, immutable=True)
    object_registry.register(TabularConvertible, test_name, test_ds, overwrite=True, immutable=True)
    object_registry.register(Code, "dataset_splitting_code", Code(splitting_code), overwrite=True)

    # Store dataset sizes
    dataset_sizes["train"].append(len(train_ds))
    dataset_sizes["validation"].append(len(val_ds))
    dataset_sizes["test"].append(len(test_ds))

    logger.debug(
        f"✅ Registered custom split of dataset {dataset_name} into train/validation/test with sizes "
        f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
    )

    return {
        "training_dataset": train_name,
        "validation_dataset": val_name,
        "test_dataset": test_name,
        "dataset_size": dataset_sizes,
    }


# TODO: does not need to be a tool
@tool
def create_input_sample(n_samples: int = 5) -> bool:
    """
    Create and register a synthetic sample input dataset that matches the model's input schema.
    This sample is used for validating inference code.

    Args:
        n_samples: Number of samples to generate (default: 5)

    Returns:
        True if sample was successfully created and registered, False otherwise
    """
    object_registry = ObjectRegistry()
    input_schema = object_registry.get(dict, "input_schema")

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
        object_registry.register(list, "predictor_input_sample", input_sample_dicts, overwrite=True, immutable=True)
        logger.debug(
            f"✅ Registered synthetic input sample with {len(input_sample_dicts)} examples for inference validation"
        )
        return True

    except Exception as e:
        logger.warning(f"⚠️ Error creating input sample for validation: {str(e)}")
        return False


@tool
def drop_null_columns(dataset_name: str) -> str:
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
        object_registry.register(TabularConvertible, dataset_name, DatasetAdapter.coerce(df), immutable=True)

        return f"Successfully dropped {n_dropped} null columns from dataset '{dataset_name}'"

    except Exception as e:
        raise RuntimeError(f"Failed to drop null columns from dataset '{dataset_name}': {str(e)}")


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
) -> str:
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
        A string indicating success or failure of the registration
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

        # TODO: separate EDA reports for raw and transformed data
        # Register in registry
        object_registry.register(dict, f"eda_report_{dataset_name}", eda_report, overwrite=True)
        logger.debug(f"✅ Registered EDA report for dataset '{dataset_name}'")
        return f"Successfully registered EDA report for dataset '{dataset_name}'"

    except Exception as e:
        logger.warning(f"⚠️ Error registering EDA report: {str(e)}")
        raise RuntimeError(f"Failed to register EDA report for dataset '{dataset_name}': {str(e)}")


@tool
def register_feature_engineering_report(
    dataset_name: str,
    overview: Dict[str, Any],
    feature_catalog: Dict[str, Any],
    feature_importance: Dict[str, Any],
    insights: List[str],
    recommendations: List[str],
) -> str:
    """
    Register a feature engineering report for a transformed dataset. This tool registers a structured report with
    actionable insights from feature engineering for use by other agents. The purpose is to ensure that the features
    created during feature engineering are well-documented.

    Args:
        dataset_name: Name of the dataset that was analyzed
        overview: Essential dataset statistics including target variable analysis
        feature_catalog: Catalog of engineered features with descriptions and transformations
        feature_importance: Assessment of feature predictive potential and relevance
        insights: Key insights derived from the analysis that directly impact feature engineering
        recommendations: Specific, prioritized actions for preprocessing and feature engineering

    Returns:
        A string indicating success or failure of the registration
    """
    object_registry = ObjectRegistry()

    try:
        # Create structured feature engineering report
        fe_report = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overview": overview,
            "feature_catalog": feature_catalog,
            "feature_importance": feature_importance,
            "insights": insights,
            "recommendations": recommendations,
        }

        # Register in registry
        object_registry.register(dict, f"fe_report_{dataset_name}", fe_report, overwrite=True)
        logger.debug(f"✅ Registered Feature Engineering report for dataset '{dataset_name}'")
        return f"Successfully registered Feature Engineering report for dataset '{dataset_name}'"

    except Exception as e:
        logger.warning(f"⚠️ Error registering Feature Engineering report: {str(e)}")
        raise RuntimeError(f"Failed to register Feature Engineering report for dataset '{dataset_name}': {str(e)}")


@tool
def get_latest_datasets() -> Dict[str, str]:
    """
    Get the most recent version of each dataset in the pipeline. Automatically detects transformed
    versions and returns the latest. Use this tool to recall what datasets are available.

    Returns:
        Dictionary mapping dataset roles to actual dataset names:
        - "raw": The original dataset
        - "transformed": The transformed dataset (if available)
        - "train": Training split (transformed version if available)
        - "val": Validation split (transformed version if available)
        - "test": Test split (transformed version if available)
    """
    object_registry = ObjectRegistry()

    try:
        all_datasets = object_registry.list_by_type(TabularConvertible)
        if not all_datasets:
            return {}

        result = {}

        # Find raw datasets (no suffixes)
        raw_datasets = [
            d for d in all_datasets if not any(suffix in d for suffix in ["_train", "_val", "_test", "_transformed"])
        ]
        if raw_datasets:
            # Use the first one (could be enhanced to handle multiple)
            result["raw"] = raw_datasets[0]

        # Find transformed dataset (not a split)
        transformed = [
            d
            for d in all_datasets
            if d.endswith("_transformed")
            and not any(d.endswith(f"_transformed_{split}") for split in ["train", "val", "test"])
        ]
        if transformed:
            result["transformed"] = transformed[0]

        # Find splits - prefer transformed versions
        for split in ["train", "val", "test"]:
            # First look for transformed split
            transformed_split = [d for d in all_datasets if d.endswith(f"_transformed_{split}")]
            if transformed_split:
                result[split] = transformed_split[0]
                continue

            # Fall back to regular split
            regular_split = [d for d in all_datasets if d.endswith(f"_{split}") and "_transformed_" not in d]
            if regular_split:
                result[split] = regular_split[0]

        return result

    except Exception as e:
        logger.warning(f"⚠️ Error getting latest datasets: {str(e)}")
        return {}


@tool
def get_dataset_for_splitting() -> str:
    """
    Get the most appropriate dataset for splitting. Returns transformed version if available,
    otherwise raw. Use this tool to get the dataset that needs to be split.

    Returns:
        Name of the dataset to split

    Raises:
        ValueError: If no suitable dataset is found for splitting
    """
    object_registry = ObjectRegistry()

    try:
        all_datasets = object_registry.list_by_type(TabularConvertible)

        # First, check if splits already exist
        has_splits = any(d.endswith(("_train", "_val", "_test")) for d in all_datasets)

        # Prefer transformed datasets that haven't been split yet
        transformed_unsplit = [
            d
            for d in all_datasets
            if d.endswith("_transformed")
            and not any(f"{d}_{split}" in all_datasets for split in ["train", "val", "test"])
        ]
        if transformed_unsplit:
            # Return the most recent (last) one
            return transformed_unsplit[-1]

        # If no unsplit transformed datasets, check for raw datasets that haven't been split
        raw_unsplit = [
            d
            for d in all_datasets
            if not any(suffix in d for suffix in ["_train", "_val", "_test", "_transformed"])
            and not any(f"{d}_{split}" in all_datasets for split in ["train", "val", "test"])
        ]

        if raw_unsplit:
            return raw_unsplit[-1]

        # If everything has been split, raise an informative error
        if has_splits:
            raise ValueError("All datasets have already been split. No unsplit datasets available.")
        else:
            raise ValueError("No datasets available for splitting. Ensure datasets have been registered.")

    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        logger.warning(f"⚠️ Error finding dataset for splitting: {str(e)}")
        raise ValueError(f"Failed to find dataset for splitting: {str(e)}")


@tool
def get_training_datasets() -> Dict[str, str]:
    """
    Get datasets ready for model training.
    Automatically finds the best available train/validation datasets.

    Returns:
        Dictionary with 'train' and 'validation' dataset names

    Raises:
        ValueError: If training datasets are not found
    """
    object_registry = ObjectRegistry()

    try:
        all_datasets = object_registry.list_by_type(TabularConvertible)

        # Look for train/val pairs, preferring transformed versions
        train_datasets = []
        val_datasets = []

        # First try to find transformed splits
        for d in all_datasets:
            if d.endswith("_transformed_train"):
                train_datasets.append((d, 1))  # Priority 1 for transformed
            elif d.endswith("_train") and "_transformed_" not in d:
                train_datasets.append((d, 2))  # Priority 2 for regular
            elif d.endswith("_transformed_val"):
                val_datasets.append((d, 1))
            elif d.endswith("_val") and "_transformed_" not in d:
                val_datasets.append((d, 2))

        # Sort by priority (lower is better)
        train_datasets.sort(key=lambda x: x[1])
        val_datasets.sort(key=lambda x: x[1])

        if not train_datasets or not val_datasets:
            raise ValueError("Training datasets not found. Ensure datasets have been split into train/validation sets.")

        # Return the best available pair
        return {"train": train_datasets[0][0], "validation": val_datasets[0][0]}

    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        logger.warning(f"⚠️ Error getting training datasets: {str(e)}")
        raise ValueError(f"Failed to get training datasets: {str(e)}")


@tool
def get_test_dataset() -> str:
    """
    Get the name of the test dataset for final model evaluation.

    Returns:
        Name of the test dataset

    Raises:
        ValueError: If test dataset is not found
    """
    object_registry = ObjectRegistry()

    try:
        all_datasets = object_registry.list_by_type(TabularConvertible)

        # Look for test datasets, preferring transformed version
        test_datasets = []

        for d in all_datasets:
            if d.endswith("_transformed_test"):
                test_datasets.append((d, 1))  # Priority 1 for transformed
            elif d.endswith("_test") and "_transformed_" not in d:
                test_datasets.append((d, 2))  # Priority 2 for regular

        if not test_datasets:
            raise ValueError("Test dataset not found. Ensure datasets have been split into train/validation/test sets.")

        # Sort by priority and return the best
        test_datasets.sort(key=lambda x: x[1])
        return test_datasets[0][0]

    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        logger.warning(f"⚠️ Error getting test dataset: {str(e)}")
        raise ValueError(f"Failed to get test dataset: {str(e)}")


# TODO: this can return a very large amount of data, consider dividing this into list_reports() and get_report(name)
@tool
def get_dataset_reports() -> Dict[str, Dict]:
    """
    Get all available data analysis reports, including EDA for raw datasets and feature engineering reports
    for transformed datasets.

    Returns:
        Dictionary with the following structure:

    """
    object_registry = ObjectRegistry()

    try:
        # Get all dict objects from registry
        all_dicts = object_registry.list_by_type(dict)

        # Filter for EDA reports (they have pattern "eda_report_{dataset_name}")
        eda_reports = {}
        for name in all_dicts:
            if name.startswith("eda_report_"):
                # Extract dataset name
                dataset_name = name[11:]  # Remove "eda_report_" prefix
                try:
                    report = object_registry.get(dict, name)
                    eda_reports[dataset_name] = report
                except Exception as e:
                    logger.debug(f"Failed to retrieve EDA report {name}: {str(e)}")
                    continue

        # Filter for feature engineering reports (they have pattern "fe_report_{dataset_name}")
        fe_reports = {}
        for name in all_dicts:
            if name.startswith("fe_report_"):
                # Extract dataset name
                dataset_name = name[10:]  # Remove "fe_report_" prefix
                try:
                    report = object_registry.get(dict, name)
                    fe_reports[dataset_name] = report
                except Exception as e:
                    logger.debug(f"Failed to retrieve Feature Engineering report {name}: {str(e)}")
                    continue

        return {
            "eda_reports": eda_reports,
            "feature_engineering_reports": fe_reports,
        }

    except Exception as e:
        logger.warning(f"⚠️ Error getting EDA reports: {str(e)}")
        return {}
