"""
Tools for schema inference, definition, and validation.
"""

import json
import logging
from typing import Dict, List, Any

import pandas as pd
from smolagents import tool

from plexe.config import prompt_templates
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.provider import Provider
from plexe.internal.common.registries.objects import ObjectRegistry

logger = logging.getLogger(__name__)


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


def get_model_schema_resolver(llm_to_use: str):
    """Returns a tool function to define model schemas with the model ID pre-filled."""

    @tool
    def define_model_schemas(
        intent: str,
        dataset_names: List[str],
        user_input_schema: Dict[str, str] = None,
        user_output_schema: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Define or validate input and output schemas for an ML model.

        This tool handles two scenarios:
        1. Schema Inference: If user schemas are not provided, infer them based on intent and data
        2. Schema Validation: If user schemas are provided, validate them and determine any needed transformations

        Args:
            intent: Natural language description of the model's purpose
            dataset_names: List of dataset registry names available to the model
            user_input_schema: Optional user-provided input schema as field:type dict
            user_output_schema: Optional user-provided output schema as field:type dict

        Returns:
            Dictionary containing:
            - input_schema: The finalized input schema
            - output_schema: The finalized output schema
            - reasoning: Explanation of schema design decisions
        """
        provider = Provider(llm_to_use)
        object_registry = ObjectRegistry()

        # Get raw dataset information for context
        dataset_previews = []
        for name in dataset_names:
            try:
                dataset = object_registry.get(TabularConvertible, name)
                df = dataset.to_pandas().head(5)
                preview = {
                    "name": name,
                    "sample": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                    "types": {col: str(df[col].dtype) for col in df.columns},
                }
                dataset_previews.append(preview)
            except Exception as e:
                logger.warning(f"Error getting preview for dataset {name}: {e}")

        # Create prompt for schema resolution
        if user_input_schema is None and user_output_schema is None:
            # Schema inference mode
            prompt = f"""
            Based on this ML task intent: "{intent}"
            
            And these raw datasets:
            {json.dumps(dataset_previews, indent=2)}
            
            Define the optimal input and output schemas for the model.
            
            IMPORTANT GUIDELINES:
            1. The model schema might need to DIFFER from raw data schemas if transformation is needed
            2. Focus on creating schemas that directly serve the model's purpose 
            3. Return schemas using only these Python types: "int", "float", "str", "bool"
            4. Provide clear reasoning for your schema design decisions
            
            Return a JSON object with these properties:
            - input_schema: dictionary mapping field names to types
            - output_schema: dictionary mapping field names to types  
            - reasoning: explanation of your schema design decisions
            """

            # Use schema prompt templates if available
            if hasattr(prompt_templates, "schema_resolve"):
                prompt = prompt_templates.schema_resolve(
                    intent=intent,
                    datasets=json.dumps(dataset_previews, indent=2),
                    has_input_schema=False,
                    has_output_schema=False,
                )
        else:
            # Schema validation mode
            prompt = f"""
            Based on this ML task intent: "{intent}"
            
            And these raw datasets:
            {json.dumps(dataset_previews, indent=2)}
            
            User-provided input schema: {json.dumps(user_input_schema or {}, indent=2)}
            User-provided output schema: {json.dumps(user_output_schema or {}, indent=2)}
            
            Validate these schemas against the raw data and determine if transformations are needed.
            
            IMPORTANT GUIDELINES:
            1. The model schema might need to DIFFER from raw data schemas if transformation is needed
            2. Focus on creating schemas that directly serve the model's purpose
            3. For any missing schema (input or output), infer it to complement what was provided
            4. Provide clear reasoning for your schema validation and any modifications
            
            Return a JSON object with these properties:
            - input_schema: finalized input schema (dictionary mapping field names to types)  
            - output_schema: finalized output schema (dictionary mapping field names to types)
            - reasoning: explanation of schema validation decisions
            """

            # Use schema prompt templates if available
            if hasattr(prompt_templates, "schema_validate"):
                prompt = prompt_templates.schema_validate(
                    intent=intent,
                    datasets=json.dumps(dataset_previews, indent=2),
                    input_schema=json.dumps(user_input_schema or {}, indent=2),
                    output_schema=json.dumps(user_output_schema or {}, indent=2),
                    has_input_schema=user_input_schema is not None,
                    has_output_schema=user_output_schema is not None,
                )

        # Query the LLM
        response_text = provider.query(prompt)

        try:
            # Parse the JSON response
            response = json.loads(response_text)

            # Ensure required fields are present
            if "input_schema" not in response or "output_schema" not in response:
                raise ValueError("LLM response missing required schema fields")

            # Register the schemas in the registry for other tools to use
            object_registry.register(dict, "input_schema", response["input_schema"])
            object_registry.register(dict, "output_schema", response["output_schema"])

            # Also register the reasoning, which could be useful for the ML Engineer agent
            if "reasoning" in response:
                object_registry.register(str, "schema_reasoning", response["reasoning"])

            return response

        except Exception as e:
            logger.error(f"Error parsing LLM response for schema resolution: {e}")
            raise

    return define_model_schemas
