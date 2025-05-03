"""
Tools related to code validation, including syntax and security checks.
"""

import logging
import uuid
from typing import Dict, List

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
        A dictionary containing validation results
    """
    validator = TrainingCodeValidator()
    validation = validator.validate(training_code)

    if validation.passed:
        return _success_response(validation.message)
    else:
        error_type = type(validation.exception).__name__ if validation.exception else "UnknownError"
        error_details = str(validation.exception) if validation.exception else "Unknown error"
        return _error_response("validation", error_type, error_details, validation.message)


@tool
def validate_inference_code(
    inference_code: str,
    model_artifact_names: List[str],
) -> Dict:
    """
    Validates inference code for syntax, security, and correctness.

    Args:
        inference_code: The inference code to validate
        model_artifact_names: Names of model artifacts to use from registry

    Returns:
        Dict with validation results and error details if validation fails
    """
    from plexe.internal.common.utils.pydantic_utils import map_to_basemodel
    from plexe.internal.common.registries.objects import ObjectRegistry

    object_registry = ObjectRegistry()

    # Get schemas from registry
    try:
        input_schema = object_registry.get(dict, "input_schema")
        output_schema = object_registry.get(dict, "output_schema")
    except Exception as e:
        return _error_response("schema_preparation", type(e).__name__, str(e))

    # Convert schemas to pydantic models
    try:
        input_model = map_to_basemodel("InputSchema", input_schema)
        output_model = map_to_basemodel("OutputSchema", output_schema)
    except Exception as e:
        return _error_response("schema_preparation", type(e).__name__, str(e))

    # Get input samples
    try:
        input_samples = object_registry.get(list, "predictor_input_sample")
        if not input_samples:
            return _error_response("input_sample", "MissingData", "Input sample list is empty")
    except Exception as e:
        return _error_response("input_sample", type(e).__name__, str(e))

    # Get artifacts
    artifact_objects = []
    try:
        for name in model_artifact_names:
            try:
                artifact_objects.append(object_registry.get(Artifact, name))
            except KeyError:
                return _error_response("artifacts", "MissingArtifact", f"Artifact '{name}' not found")

        if not artifact_objects:
            return _error_response("artifacts", "NoArtifacts", "No artifacts available for model loading")
    except Exception as e:
        return _error_response("artifacts", type(e).__name__, str(e))

    # Validate the code
    validator = InferenceCodeValidator(input_schema=input_model, output_schema=output_model, input_sample=input_samples)
    validation = validator.validate(inference_code, model_artifacts=artifact_objects)

    # Return appropriate result
    if validation.passed:
        inference_code_id = uuid.uuid4().hex
        object_registry.register(Code, inference_code_id, Code(inference_code))
        return _success_response(validation.message, inference_code_id)

    # Extract error details from validation result
    error_type = validation.error_type or (
        type(validation.exception).__name__ if validation.exception else "UnknownError"
    )
    error_details = validation.error_details or (str(validation.exception) if validation.exception else "Unknown error")

    return _error_response(validation.error_stage or "unknown", error_type, error_details, validation.message)


def _error_response(stage, exc_type, details, message=None):
    """Helper to create error response dictionaries"""
    return {
        "passed": False,
        "error_stage": stage,
        "error_type": exc_type,
        "error_details": details,
        "message": message or details,
    }


def _success_response(message, inference_code_id=None):
    """Helper to create success response dictionaries"""
    response = {"passed": True, "message": message}
    # Only include inference_code_id for inference code validation
    if inference_code_id is not None:
        response["inference_code_id"] = inference_code_id
    return response
