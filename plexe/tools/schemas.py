"""
Tools for schema inference, definition, and validation.
"""

import logging
from typing import Dict, Any

import pandas as pd
from smolagents import tool

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.utils.pydantic_utils import map_to_basemodel

logger = logging.getLogger(__name__)


@tool
def register_final_model_schemas(
    input_schema: Dict[str, str], output_schema: Dict[str, str], reasoning: str
) -> Dict[str, str]:
    """
    Register agent-determined schemas in the ObjectRegistry with validation.

    Validates schemas by attempting to convert them to Pydantic models
    and registers them in the ObjectRegistry if valid. If validation fails,
    raises an exception with details.

    Args:
        input_schema: Finalized input schema as field:type dictionary
        output_schema: Finalized output schema as field:type dictionary
        reasoning: Explanation of schema design decisions

    Returns:
        Status message confirming registration

    Raises:
        ValueError: If schema validation fails
        KeyError: If schema registration fails
    """
    object_registry = ObjectRegistry()

    # Validate schemas by attempting to convert them to Pydantic models
    try:
        map_to_basemodel("InputSchema", input_schema)
        map_to_basemodel("OutputSchema", output_schema)
    except Exception as e:
        error_msg = f"Schema validation or registration failed: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    # Register input schema if possible
    try:
        object_registry.register(dict, "input_schema", input_schema)
    except ValueError as e:
        if "already registered" not in str(e):
            raise e

    # Register output schema if possible
    try:
        object_registry.register(dict, "output_schema", output_schema)
    except ValueError as e:
        if "already registered" not in str(e):
            raise e

    # Register reasoning if possible
    try:
        object_registry.register(str, "schema_reasoning", reasoning)
    except ValueError as e:
        if "already registered" not in str(e):
            raise e

    return {"status": "success", "message": "Schemas validated and registered successfully"}


@tool
def get_dataset_schema(dataset_name: str) -> Dict[str, Any]:
    """
    Extract the schema (column names and types) from a dataset. This is useful for understanding the structure
    of the dataset and how it can be used in model training.

    Args:
        dataset_name: Name of the dataset in the registry

    Returns:
        Dictionary with column names and their python types
    """
    object_registry = ObjectRegistry()
    dataset = object_registry.get(TabularConvertible, dataset_name)
    df = dataset.to_pandas()

    # Get column names and infer python types
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        # Map pandas types to Python types
        if pd.api.types.is_integer_dtype(dtype):
            py_type = "int"
        elif pd.api.types.is_float_dtype(dtype):
            py_type = "float"
        elif pd.api.types.is_bool_dtype(dtype):
            py_type = "bool"
        else:
            py_type = "str"
        schema[col] = py_type

    return {"dataset_name": dataset_name, "columns": schema}


@tool
def get_model_schemas() -> Dict[str, Dict[str, str]]:
    """
    Get input and output schemas if available.

    Returns:
        Dictionary with 'input' and 'output' schemas (if registered).
        Each schema is a dict mapping field names to types.
        Returns empty dict for missing schemas.
    """
    object_registry = ObjectRegistry()
    result = {}

    try:
        # Try to get input schema
        try:
            input_schema = object_registry.get(dict, "input_schema")
            if input_schema:
                result["input"] = input_schema
        except KeyError:
            logger.debug("Input schema not found in registry")

        # Try to get output schema
        try:
            output_schema = object_registry.get(dict, "output_schema")
            if output_schema:
                result["output"] = output_schema
        except KeyError:
            logger.debug("Output schema not found in registry")

        return result

    except Exception as e:
        logger.warning(f"⚠️ Error getting model schemas: {str(e)}")
        return {}
