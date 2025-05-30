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
def register_global_schemas(
    input_schema: Dict[str, str], output_schema: Dict[str, str], reasoning: str
) -> Dict[str, str]:
    """
    Register input and output schemas that should be used by all models built for all solutions.

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

    # Register input schema if possible; global schemas are typically registered once
    try:
        object_registry.register(dict, "input_schema", input_schema, immutable=True)
    except ValueError as e:
        if "already registered" not in str(e):
            raise e

    # Register output schema if possible; global schemas are typically registered once
    try:
        object_registry.register(dict, "output_schema", output_schema, immutable=True)
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
def get_global_schemas() -> Dict[str, Dict[str, str]]:
    """
    Get global input and output schemas that should apply to a model.

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
            logger.debug("Global input schema not found in registry")

        # Try to get output schema
        try:
            output_schema = object_registry.get(dict, "output_schema")
            if output_schema:
                result["output"] = output_schema
        except KeyError:
            logger.debug("Global output schema not found in registry")

        return result

    except Exception as e:
        logger.warning(f"⚠️ Error getting global schemas: {str(e)}")
        return {}


@tool
def register_solution_schemas(
    solution_id: str, input_schema: Dict[str, str], output_schema: Dict[str, str], reasoning: str
) -> Dict[str, str]:
    """
    Register input and output schemas for a specific solution.

    Args:
        solution_id: ID of the solution to register schemas for
        input_schema: Solution-specific input schema as field:type dictionary
        output_schema: Solution-specific output schema as field:type dictionary
        reasoning: Explanation of schema design decisions

    Returns:
        Status message confirming registration

    Raises:
        ValueError: If schema validation fails or solution not found
    """
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()

    # If schemas are locked by the user, we must use the global schemas
    input_is_locked = object_registry.get(bool, "input_schema_is_locked")
    output_is_locked = object_registry.get(bool, "output_schema_is_locked")
    global_schemas = get_global_schemas()
    global_input_schema = global_schemas.get("input")
    global_output_schema = global_schemas.get("output")

    solution = object_registry.get(Solution, solution_id)

    # If both schemas are locked, use global schemas
    if input_is_locked and output_is_locked:
        solution.input_schema = global_input_schema
        solution.output_schema = global_output_schema
        solution.schema_reasoning = "Using schemas provided by the user"
    # If input is locked, validate output schema and set
    elif input_is_locked:
        try:
            map_to_basemodel("OutputSchema", output_schema)
        except Exception as e:
            error_msg = f"Output schema validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        solution.input_schema = global_input_schema
        solution.output_schema = output_schema
        solution.schema_reasoning = reasoning
    # If output is locked, validate input schema and set
    elif output_is_locked:
        try:
            map_to_basemodel("InputSchema", input_schema)
        except Exception as e:
            error_msg = f"Input schema validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        solution.input_schema = input_schema
        solution.output_schema = global_output_schema
        solution.schema_reasoning = reasoning
    # If neither schema is locked, validate both schemas
    else:
        try:
            map_to_basemodel("InputSchema", input_schema)
            map_to_basemodel("OutputSchema", output_schema)
        except Exception as e:
            error_msg = f"Schema validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        solution.input_schema = input_schema
        solution.output_schema = output_schema
        solution.schema_reasoning = reasoning

    # Re-register the updated solution
    object_registry.register(Solution, solution_id, solution, overwrite=True)

    # Construct response depending on actions taken
    if input_is_locked and output_is_locked:
        logger.debug(f"✅ Registered global schemas for solution '{solution_id}' (schemas locked)")
        return {
            "status": "success",
            "message": "Nothing to register, as schemas are locked by user.",
            "registered_input_schema": global_input_schema,
            "registered_output_schema": global_output_schema,
        }
    elif input_is_locked:
        logger.debug(f"✅ Registered output schema for solution '{solution_id}' (input locked)")
        return {
            "status": "success",
            "message": "New output schema was registered; input schema is locked by the user so defaulted to global.",
            "registered_input_schema": global_input_schema,
            "registered_output_schema": output_schema,
        }
    elif output_is_locked:
        logger.debug(f"✅ Registered input schema for solution '{solution_id}' (output locked)")
        return {
            "status": "success",
            "message": "New input schema was registered; output schema is locked by the user so defaulted to global.",
            "registered_input_schema": input_schema,
            "registered_output_schema": global_output_schema,
        }
    else:
        logger.debug(f"✅ Registered schemas for solution '{solution_id}'")
        return {
            "status": "success",
            "message": f"Schemas validated and registered for solution '{solution_id}'",
            "registered_input_schema": input_schema,
            "registered_output_schema": output_schema,
        }


@tool
def get_solution_schemas(solution_id: str) -> Dict[str, Dict[str, str]]:
    """
    Get schemas for a specific solution, with fallback to global schemas.

    Args:
        solution_id: ID of the solution to get schemas for

    Returns:
        Dictionary with 'input' and 'output' schemas.
        Prioritizes solution-specific schemas over global schemas.
        Returns empty dict if no schemas found.
    """
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()
    result = {}

    try:
        # First try to get solution-specific schemas
        try:
            solution = object_registry.get(Solution, solution_id)
            if solution.input_schema:
                result["input"] = solution.input_schema
            if solution.output_schema:
                result["output"] = solution.output_schema

            # If we have both schemas from solution, return them
            if "input" in result and "output" in result:
                logger.debug(f"Using solution-specific schemas for '{solution_id}'")
                return result

        except KeyError:
            logger.debug(f"Solution '{solution_id}' not found, falling back to global schemas")

        # Fallback to global schemas for missing schemas
        global_schemas = get_global_schemas()
        if "input" not in result and "input" in global_schemas:
            result["input"] = global_schemas["input"]
        if "output" not in result and "output" in global_schemas:
            result["output"] = global_schemas["output"]

        return result

    except Exception as e:
        logger.warning(f"⚠️ Error getting solution schemas: {str(e)}")
        return {}
