"""
Tools related to code validation, including syntax and security checks.
"""

import logging
import uuid
import ast
from typing import Dict, List

from smolagents import tool

from plexe.config import code_templates
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
    from plexe.core.object_registry import ObjectRegistry

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

        # Also instantiate and register the predictor for the model tester agent
        try:
            import types

            predictor_module = types.ModuleType("predictor")
            exec(inference_code, predictor_module.__dict__)
            predictor_class = getattr(predictor_module, "PredictorImplementation")
            predictor = predictor_class(artifact_objects)

            # Register the instantiated predictor
            from plexe.core.interfaces.predictor import Predictor

            object_registry.register(Predictor, "trained_predictor", predictor, overwrite=True)
            logger.debug("✅ Registered instantiated predictor for testing")

        except Exception as e:
            logger.warning(f"⚠️ Failed to register instantiated predictor: {str(e)}")
            # Don't fail validation if predictor registration fails

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


@tool
def validate_feature_transformations(transformation_code: str) -> Dict:
    """
    Validates feature transformation code for syntax correctness and implementation
    of the FeatureTransformer interface.

    Args:
        transformation_code: Python code for transforming datasets

    Returns:
        Dictionary with validation results
    """
    import types
    import warnings
    from plexe.core.object_registry import ObjectRegistry
    from plexe.core.interfaces.feature_transformer import FeatureTransformer

    # Check for syntax errors
    try:
        ast.parse(transformation_code)
    except SyntaxError as e:
        return _error_response("syntax", "SyntaxError", str(e))

    # Load the code as a module to check for proper FeatureTransformer implementation
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = types.ModuleType("test_feature_transformer")
            exec(transformation_code, module.__dict__)

            # Check if the module contains the FeatureTransformerImplementation class
            if not hasattr(module, "FeatureTransformerImplementation"):
                return _error_response(
                    "class_definition",
                    "MissingClass",
                    "Code must define a class named 'FeatureTransformerImplementation'",
                )

            # Check if the class is a subclass of FeatureTransformer
            transformer_class = getattr(module, "FeatureTransformerImplementation")
            if not issubclass(transformer_class, FeatureTransformer):
                return _error_response(
                    "class_definition",
                    "InvalidClass",
                    "FeatureTransformerImplementation must be a subclass of FeatureTransformer",
                )
    except Exception as e:
        return _error_response(
            "validation",
            type(e).__name__,
            str(e),
            message=f"The feature transformer must be a subclass of the following interface:\n\n"
            f"```python\n"
            f"{code_templates.feature_transformer_interface}"
            f"```",
        )

    # Register the transformation code with a fixed ID
    object_registry = ObjectRegistry()
    code_id = "feature_transformations"
    object_registry.register(Code, code_id, Code(transformation_code), overwrite=True)

    return {"passed": True, "message": "Feature transformation code validated successfully", "code_id": code_id}
