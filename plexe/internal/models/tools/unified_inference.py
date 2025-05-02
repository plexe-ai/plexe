"""
Unified tools for generating and validating inference code.
"""

import logging
import uuid
import json
from typing import Dict, Any, Callable

from smolagents import tool

from plexe.config import code_templates, prompt_templates
from plexe.internal.common.provider import Provider
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.common.utils.pydantic_utils import map_to_basemodel
from plexe.internal.common.utils.response import extract_code
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.code import Code
from plexe.internal.models.validation.composites import InferenceCodeValidator

import pandas as pd

logger = logging.getLogger(__name__)


def get_generate_and_validate_inference_code(llm_to_use: str) -> Callable:
    """Returns a tool function that generates and validates inference code with the model ID pre-filled."""

    @tool
    def generate_and_validate_inference_code(
        input_schema: Dict[str, str], output_schema: Dict[str, str], training_code_id: str, instructions: str = ""
    ) -> Dict[str, Any]:
        """
        Generates and validates inference code in a single operation.

        Args:
            input_schema: The input schema for the model, for example {"feat_1": "int", "feat_2": "str"}
            output_schema: The output schema for the model, for example {"output": "float"}
            training_code_id: The identifier for the training code used to train the model
            instructions: Optional instructions from the agent to guide code generation

        Returns:
            A dictionary containing validation results and code information
        """

        # Retrieve training code
        try:
            training_code = ObjectRegistry().get(Code, training_code_id).code
            logger.debug(f"Retrieved training code with ID {training_code_id}")
        except Exception as e:
            return {
                "passed": False,
                "message": f"Training code with ID {training_code_id} not found",
                "exception": str(e),
                "code": None,
            }

        # Convert dict schemas to pydantic models
        try:
            input_model = map_to_basemodel("InputSchema", input_schema)
            output_model = map_to_basemodel("OutputSchema", output_schema)
        except Exception as e:
            return {
                "passed": False,
                "message": "Failed to convert schemas to pydantic models",
                "exception": str(e),
                "code": None,
            }

        # Generate inference code
        try:
            provider = Provider(llm_to_use)
            response = provider.query(
                system_message=prompt_templates.inference_system(),
                user_message=prompt_templates.inference_unified_generate(
                    predictor_interface_source=code_templates.predictor_interface,
                    predictor_template=code_templates.predictor_template,
                    training_code=training_code,
                    input_schema=json.dumps(input_schema, indent=2),
                    output_schema=json.dumps(output_schema, indent=2),
                    instructions=instructions,
                ),
            )
            inference_code = extract_code(response)
            logger.debug("Generated inference code successfully")
        except Exception as e:
            return {"passed": False, "message": "Failed to generate inference code", "exception": str(e), "code": None}

        # Retrieve input sample from registry for validation
        try:
            input_df = ObjectRegistry().get(pd.DataFrame, "predictor_input_sample")
            logger.debug(f"Retrieved input sample for validation with {len(input_df)} rows")
        except Exception as e:
            return {
                "passed": False,
                "message": "Failed to retrieve input sample for validation",
                "exception": str(e),
                "code": inference_code,
            }

        # Retrieve model artifacts
        artifact_objects = []
        try:
            # Get all artifact objects from registry
            artifacts = ObjectRegistry().get_all(Artifact)
            if artifacts:
                artifact_objects = list(artifacts.values())
                logger.debug(f"Retrieved {len(artifact_objects)} artifacts for validation")
        except Exception as e:
            logger.warning(f"Failed to retrieve artifacts: {str(e)}")
            # Continue without artifacts - validation will likely fail but we should try

        # Validate the inference code
        validator = InferenceCodeValidator(
            input_schema=input_model,
            output_schema=output_model,
            input_sample=input_df,
        )

        validation = validator.validate(inference_code, model_artifacts=artifact_objects)

        result = {
            "passed": validation.passed,
            "message": validation.message,
            "exception": str(validation.exception) if validation.exception else None,
            "code": inference_code,
        }

        if validation.passed:
            # Register successful code in registry
            inference_code_id = uuid.uuid4().hex
            ObjectRegistry().register(Code, inference_code_id, Code(inference_code))
            result["inference_code_id"] = inference_code_id
            logger.debug(f"Registered validated inference code with ID {inference_code_id}")

        return result

    return generate_and_validate_inference_code
