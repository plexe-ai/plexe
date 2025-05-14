"""
Tools for schema inference, definition, and validation.
"""

import logging
from typing import Dict, Any

import pandas as pd
from smolagents import tool

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.registries.objects import ObjectRegistry

logger = logging.getLogger(__name__)


@tool
def register_final_model_schemas(
    input_schema: Dict[str, str], output_schema: Dict[str, str], reasoning: str
) -> Dict[str, str]:
    """
    Register agent-determined schemas in the ObjectRegistry.

    Args:
        input_schema: Finalized input schema as field:type dictionary
        output_schema: Finalized output schema as field:type dictionary
        reasoning: Explanation of schema design decisions

    Returns:
        Status message confirming registration
    """
    object_registry = ObjectRegistry()
    object_registry.register(dict, "input_schema", input_schema)
    object_registry.register(dict, "output_schema", output_schema)
    object_registry.register(str, "schema_reasoning", reasoning)
    return {"status": "Schemas registered successfully"}


@tool
def get_raw_dataset_schema(dataset_name: str) -> Dict[str, Any]:
    """
    Extract the schema (column names and types) from a raw dataset.

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
