"""
Tools related to code validation, including syntax and security checks.
"""

import logging
import ast
from typing import Dict

from smolagents import tool

from plexe.config import code_templates
from plexe.internal.models.entities.code import Code
from plexe.internal.models.validation.composites import (
    InferenceCodeValidator,
    TrainingCodeValidator,
)
from plexe.tools.schemas import get_solution_schemas

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
    solution_id: str,
    inference_code: str,
) -> Dict:
    """
    Validates inference code for syntax, security, and correctness, and updates the Solution object.

    Args:
        solution_id: ID of the Solution object to update with inference code
        inference_code: The inference code to validate

    Returns:
        Dict with validation results and error details if validation fails
    """
    from plexe.internal.common.utils.pydantic_utils import map_to_basemodel
    from plexe.core.object_registry import ObjectRegistry
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()

    # Get solution object from registry
    try:
        solution = object_registry.get(Solution, solution_id)
    except Exception as e:
        return _error_response("solution_retrieval", type(e).__name__, str(e))

    # Get schemas from registry
    try:
        schemas = get_solution_schemas("best_performing_solution")
        input_schema = schemas["input"]
        output_schema = schemas["output"]
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

    # Validate the code
    validator = InferenceCodeValidator(input_schema=input_model, output_schema=output_model, input_sample=input_samples)
    validation = validator.validate(inference_code, model_artifacts=solution.model_artifacts)

    # Return appropriate result
    if validation.passed:
        # Update the Solution object with inference code, and register an alias for production use
        solution.inference_code = inference_code
        object_registry.register(Solution, solution_id, solution, overwrite=True)
        object_registry.register(Solution, "final_inference_solution", solution, overwrite=True, immutable=True)

        # Also instantiate and register the predictor for the model tester agent
        try:
            import types

            predictor_module = types.ModuleType("predictor")
            exec(inference_code, predictor_module.__dict__)
            predictor_class = getattr(predictor_module, "PredictorImplementation")
            predictor = predictor_class(solution.model_artifacts)

            # Register the instantiated predictor
            from plexe.core.interfaces.predictor import Predictor

            object_registry.register(Predictor, "trained_predictor", predictor, overwrite=True)
            logger.debug("✅ Registered instantiated predictor for testing")

        except Exception as e:
            logger.warning(f"⚠️ Failed to register instantiated predictor: {str(e)}")
            # Don't fail validation if predictor registration fails

        return _success_response(validation.message, solution_id)

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


def _success_response(message, solution_id=None):
    """Helper to create success response dictionaries"""
    response = {"passed": True, "message": message}
    # Only include solution_id for inference code validation
    if solution_id is not None:
        response["solution_id"] = solution_id
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
