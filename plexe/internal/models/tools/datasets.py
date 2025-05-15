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
    is_time_series: bool = False,
    time_index_column: str = None,
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
        is_time_series: Whether the data is chronological time series data (default: False)
        time_index_column: Column name that represents the time index, required if is_time_series=True

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
        train_ds, val_ds, test_ds = dataset.split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            is_time_series=is_time_series,
            time_index_column=time_index_column,
        )

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
    feature_analysis: Dict[str, Any],
    relationships: Dict[str, Any],
    data_quality: Dict[str, Any],
    insights: List[str],
    recommendations: List[str],
) -> bool:
    """
    Register an exploratory data analysis (EDA) report for a dataset in the Object Registry.

    This tool creates a structured report with findings from exploratory data analysis and
    registers it in the Object Registry for use by other agents.

    Args:
        dataset_name: Name of the dataset that was analyzed
        overview: General dataset statistics including shape, data types, memory usage
        feature_analysis: Analysis of individual features with distributions and statistics
        relationships: Correlation analysis and feature relationships
        data_quality: Information about missing values, outliers, and data issues
        insights: Key insights derived from the analysis
        recommendations: Recommendations for preprocessing and modeling

    Returns:
        True if the report was successfully registered, False otherwise
    """
    object_registry = ObjectRegistry()

    try:
        # Create structured EDA report
        eda_report = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overview": overview,
            "feature_analysis": feature_analysis,
            "relationships": relationships,
            "data_quality": data_quality,
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
