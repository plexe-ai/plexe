"""
This module provides utility functions for manipulating Pydantic models.
"""

from pydantic import BaseModel, create_model
from typing import Type, List, Dict, get_type_hints


def _validate_schema_types(schema: Type[BaseModel]) -> None:
    """Validate that a BaseModel schema only contains allowed types."""
    allowed_types = {int, float, str, bool, List[int], List[float], List[str], List[bool]}

    for field_name, field_info in schema.model_fields.items():
        field_type = field_info.annotation
        if field_type not in allowed_types:
            raise ValueError(
                f"Field '{field_name}' has unsupported type '{field_type}'. "
                f"Allowed types: int, float, str, bool, List[int], List[float], List[str], List[bool]"
            )


def merge_models(model_name: str, models: List[Type[BaseModel]]) -> Type[BaseModel]:
    """
    Merge multiple Pydantic models into a single model. The ordering of the list determines
    the overriding precedence of the models; the last model in the list will override any fields
    with the same name in the preceding models.

    :param model_name: The name of the new model to create.
    :param models: A list of Pydantic models to merge.
    :return: A new Pydantic model that combines the input models.
    """
    fields = dict()
    for model in models:
        for name, properties in model.model_fields.items():
            fields[name] = (properties.annotation, ... if properties.is_required() else properties.default)
    return create_model(model_name, **fields)


def create_model_from_fields(model_name: str, model_fields: dict) -> Type[BaseModel]:
    """
    Create a Pydantic model from a dictionary of fields.

    :param model_name: The name of the model to create.
    :param model_fields: A dictionary of field names to field properties.
    """
    for name, properties in model_fields.items():
        model_fields[name] = (properties.annotation, ... if properties.is_required() else properties.default)
    return create_model(model_name, **model_fields)


def map_to_basemodel(name: str, schema: dict | Type[BaseModel]) -> Type[BaseModel]:
    """
    Ensure that the schema is a Pydantic model or a dictionary, and return the model.

    :param [str] name: the name to be given to the model class
    :param [dict] schema: the schema to be converted to a Pydantic model
    :return: the Pydantic model
    """
    # Pydantic model: validate and return
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        _validate_schema_types(schema)
        return schema

    # Dictionary: convert to Pydantic model, if possible
    if isinstance(schema, dict):
        try:
            # Handle both Dict[str, type] and Dict[str, str] formats
            annotated_schema = {}

            for k, v in schema.items():
                # If v is a string like "int", convert it to the actual type
                if isinstance(v, str):
                    type_mapping = {
                        "int": int,
                        "float": float,
                        "str": str,
                        "bool": bool,
                        "List[int]": List[int],
                        "List[float]": List[float],
                        "List[str]": List[str],
                        "List[bool]": List[bool],
                    }
                    if v in type_mapping:
                        annotated_schema[k] = (type_mapping[v], ...)
                    else:
                        raise ValueError(f"Invalid type specification: {v} for field {k}")
                # If v is already a type or one of our allowed typing generics, use it directly
                elif isinstance(v, type) or v in {List[int], List[float], List[str], List[bool]}:
                    # Validate that it's one of our allowed types
                    allowed_types = {int, float, str, bool, List[int], List[float], List[str], List[bool]}
                    if v not in allowed_types:
                        raise ValueError(f"Unsupported type '{v}' for field '{k}'. Allowed types: {allowed_types}")
                    annotated_schema[k] = (v, ...)
                else:
                    raise ValueError(f"Invalid field specification for {k}: {v}")

            return create_model(name, **annotated_schema)
        except Exception as e:
            raise ValueError(f"Invalid schema definition: {e}")

    # All other schema types are invalid
    raise TypeError("Schema must be a Pydantic model or a dictionary of field names to types.")


def format_schema(schema: Type[BaseModel]) -> Dict[str, str]:
    """
    Format a schema model into a dictionary representation of field names and types.

    :param schema: A pydantic model defining a schema
    :return: A dictionary representing the schema structure with field names as keys and types as values
    """
    if not schema:
        return {}

    result = {}
    # Use model_fields which is the recommended approach in newer Pydantic versions
    for field_name, field_info in schema.model_fields.items():
        field_type = getattr(field_info.annotation, "__name__", str(field_info.annotation))
        result[field_name] = field_type

    return result


def convert_schema_to_type_dict(schema: Type[BaseModel]) -> Dict[str, type]:
    """
    Convert a Pydantic model to a dictionary mapping field names to their Python types.

    This is useful for tools that require type information without the full Pydantic field metadata.

    :param schema: A Pydantic model to convert
    :return: A dictionary with field names as keys and Python types as values
    """
    if not schema or not issubclass(schema, BaseModel):
        raise TypeError("Schema must be a Pydantic BaseModel")

    result = {}

    # Get the actual type annotations, which will be Python types
    type_hints = get_type_hints(schema)

    # Extract annotations from model fields
    for field_name, field_info in schema.model_fields.items():
        # Use the type hint if available, otherwise fall back to the field annotation
        field_type = type_hints.get(field_name, field_info.annotation)
        result[field_name] = field_type

    return result
