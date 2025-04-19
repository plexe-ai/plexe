"""
Tools related to code validation, including syntax and security checks.
"""

import logging
import uuid
from typing import Dict, List

import pandas as pd
from smolagents import tool

from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.code import Code
from plexe.internal.models.validation.composites import (
    InferenceCodeValidator,
    TrainingCodeValidator,
)

logger = logging.getLogger(__name__)


@tool
def validate_training_code(training_code: str) -> Dict:
    """Validates training code for syntax and security issues.

    Args:
        training_code: The training code to validate

    Returns:
        A dictionary containing validation results, in form
    """
    validator = TrainingCodeValidator()
    validation = validator.validate(training_code)
    return {
        "passed": validation.passed,
        "message": validation.message,
        "exception": str(validation.exception) if validation.exception else None,
    }


@tool
def validate_inference_code(
    inference_code: str,
    model_artifact_names: List[str],
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
) -> Dict:
    """
    Validates inference code for syntax, security, and correctness. The schemas must be provided as a flat dictionary
    mapping field names to strings representing their types (e.g., "int", "str").

    Args:
        inference_code: The inference code to validate
        model_artifact_names: The names of model artifacts to use from the registry
        input_schema: The input schema for the model, for example {"feat_1": "int", "feat_2": "str"}
        output_schema: The output schema for the model, for example {"output": "float"}

    Returns:
        A dictionary containing validation results
    """
    from plexe.internal.common.utils.pydantic_utils import map_to_basemodel
    from plexe.internal.common.registries.objects import ObjectRegistry

    # Debug logging
    logger.debug(f"Input schema type: {type(input_schema)}, value: {input_schema}")
    logger.debug(f"Output schema type: {type(output_schema)}, value: {output_schema}")

    # Convert dict schemas back to Type[BaseModel] as this is what the validator expects
    try:
        input_model = map_to_basemodel("InputSchema", input_schema)
        output_model = map_to_basemodel("OutputSchema", output_schema)
    except Exception as e:
        raise ValueError(f"❌ Given schema is not convertible to pydantic BaseModel: {str(e)}") from e

    # Initialise registries which are used to pass datasets and artifacts between agents and tools
    object_registry = ObjectRegistry()

    # Retrieve input sample from registry and convert it to a DataFrame
    try:
        input_df = object_registry.get(pd.DataFrame, "predictor_input_sample")
    except Exception as e:
        raise ValueError(f"❌ Failed to get input sample from registry: {str(e)}") from e

    # Retrieve model artifacts from registry
    artifact_objects = []
    if model_artifact_names:
        for name in model_artifact_names:
            try:
                artifact = object_registry.get(Artifact, name)
                artifact_objects.append(artifact)
                logger.debug(f"Retrieved artifact '{name}' from registry")
            except KeyError as e:
                raise ValueError(f"❌ Artifact '{name}' not found in registry") from e

    if not artifact_objects:
        raise ValueError("❌ No artifacts found or created. Validation will fail.")

    # Validate the inference code
    validation = InferenceCodeValidator(
        input_schema=input_model,
        output_schema=output_model,
        input_sample=input_df,
    ).validate(inference_code, model_artifacts=artifact_objects)

    result = {
        "passed": validation.passed,
        "message": validation.message,
        "exception": str(validation.exception) if validation.exception else None,
    }

    if validation.passed:
        inference_code_id = uuid.uuid4().hex
        object_registry.register(Code, inference_code_id, Code(inference_code))
        result["inference_code_id"] = inference_code_id

    return result
