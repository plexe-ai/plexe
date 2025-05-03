"""
Tools for providing context to agents for code generation tasks.
"""

import logging
from typing import Dict, Any

from smolagents import tool

from plexe.config import code_templates
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.models.entities.code import Code

logger = logging.getLogger(__name__)


@tool
def get_inference_context(training_code_id: str) -> Dict[str, Any]:
    """
    Provides comprehensive context needed for generating inference code. Use this tool to retrieve
    a summary of the training code, schemas, expected inputs for the purpose of planning the inference
    code.

    Args:
        training_code_id: The ID of the code that was used to train the model

    Returns:
        A dictionary containing all context needed for inference code generation
    """
    object_registry = ObjectRegistry()

    # Retrieve training code
    try:
        training_code = object_registry.get(Code, training_code_id).code
        logger.debug(f"Retrieved training code with ID {training_code_id}")
    except Exception as e:
        raise ValueError(f"Training code with ID {training_code_id} not found: {str(e)}")

    # Retrieve schemas
    try:
        input_schema = object_registry.get(dict, "input_schema")
        output_schema = object_registry.get(dict, "output_schema")
    except Exception as e:
        raise ValueError(f"Failed to retrieve schemas from registry: {str(e)}")

    # Retrieve input sample
    try:
        input_sample = object_registry.get(list, "predictor_input_sample")
        logger.debug(f"Retrieved input sample with {len(input_sample)} examples")
    except Exception as e:
        raise ValueError(f"Failed to retrieve input sample: {str(e)}")

    return {
        "training_code": training_code,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "predictor_interface": code_templates.predictor_interface,
        "predictor_template": code_templates.predictor_template,
        "input_sample": input_sample,
    }
