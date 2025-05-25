"""
Tools for conversational model definition and build initiation.

These tools support the conversational agent in helping users define their ML
requirements and starting model builds when ready.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from smolagents import tool

import plexe
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.provider import ProviderConfig
from plexe.core.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


@tool
def validate_dataset_files(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Check if specified file paths can be read as datasets using pandas.

    Args:
        file_paths: List of file paths to validate

    Returns:
        Dictionary mapping file paths to validation results with status, shape, and error info
    """
    results = {}

    for file_path in file_paths:
        result = {"valid": False, "shape": None, "columns": None, "error": None}

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result["error"] = f"File does not exist: {file_path}"
                results[file_path] = result
                continue

            # Determine file type and try to read
            path_obj = Path(file_path)
            file_extension = path_obj.suffix.lower()

            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension in [".parquet", ".pq"]:
                df = pd.read_parquet(file_path)
            else:
                result["error"] = f"Unsupported file format: {file_extension}. Supported formats: .csv, .parquet"
                results[file_path] = result
                continue

            # File successfully read
            result["valid"] = True
            result["dataset_name"] = path_obj.stem

            # Register the DataFrame in object registry
            ObjectRegistry().register(
                t=TabularConvertible, name=path_obj.stem, item=DatasetAdapter.coerce(df), immutable=True
            )

        except Exception as e:
            result["error"] = str(e)

        results[file_path] = result

    return results


@tool
def initiate_model_build(
    intent: str,
    dataset_file_paths: List[str],
    input_schema: Optional[Dict] = None,
    output_schema: Optional[Dict] = None,
    n_solutions_to_try: int = 1,
) -> Dict[str, str]:
    """
    Initiate a model build by loading datasets from file paths and starting the build process.

    Args:
        intent: Natural language description of what the model should do
        dataset_file_paths: List of file paths to dataset files (CSV or Parquet)
        input_schema: The input schema for the model, as a flat field:type dictionary; leave None if not known
        output_schema: The output schema for the model, as a flat field:type dictionary; leave None if not known
        n_solutions_to_try: Number of model solutions to try, out of which the best will be selected

    Returns:
        Dictionary with build initiation status and details
    """
    try:
        # First validate all files can be read
        validation_results = validate_dataset_files(dataset_file_paths)

        # Check if any files failed validation
        failed_files = [path for path, result in validation_results.items() if not result["valid"]]
        if failed_files:
            error_details = {path: validation_results[path]["error"] for path in failed_files}
            return {
                "status": "failed",
                "message": f"Failed to read dataset files: {failed_files}",
                "errors": error_details,
            }

        # Load datasets into DataFrames
        df = None
        for file_path in dataset_file_paths:
            path_obj = Path(file_path)
            file_extension = path_obj.suffix.lower()

            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension in [".parquet", ".pq"]:
                df = pd.read_parquet(file_path)

        # Import here to avoid circular dependencies
        from plexe.model_builder import ModelBuilder

        # Create ModelBuilder instance with loaded DataFrames
        model_builder = ModelBuilder(
            provider=ProviderConfig(
                default_provider="openai/gpt-4o",
                orchestrator_provider="anthropic/claude-3-7-sonnet-20250219",
                research_provider="openai/gpt-4o",
                engineer_provider="anthropic/claude-sonnet-4-20250514",
                ops_provider="anthropic/claude-sonnet-4-20250514",
                tool_provider="openai/gpt-4o",
            ),
        )

        # Start the build process
        logger.info(f"Initiating model build with intent: {intent}")
        logger.info(f"Using dataset files: {dataset_file_paths}")

        model = model_builder.build(
            intent=intent,
            datasets=[df],  # Pass actual DataFrames instead of names
            input_schema=input_schema,
            output_schema=output_schema,
            max_iterations=n_solutions_to_try,
        )

        plexe.save_model(model, "model-from-chat.tar.gz")

        # For now, just return success status
        return {
            "status": "initiated",
            "message": f"Model build started successfully with intent: '{intent}'",
            "dataset_files": dataset_file_paths,
            "dataset_shapes": [validation_results[path]["shape"] for path in dataset_file_paths],
        }

    except Exception as e:
        logger.error(f"Failed to initiate model build: {str(e)}")
        return {"status": "failed", "message": f"Failed to start model build: {str(e)}", "error": str(e)}
